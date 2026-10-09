"""Official OmniDocBench end-to-end evaluation, run out of process.

OmniDocBench (https://github.com/opendatalab/OmniDocBench) grades a page by parsing the
predicted markdown into text blocks, display formulas and tables, matching them against the
annotated blocks, and measuring text edit distance, table TEDS, formula CDM and
reading-order edit distance. The matching alone is several thousand lines and changes
between benchmark versions, so rather than port it this module runs the pinned official
evaluator of each version: its scores are that leaderboard's by construction.

The evaluator is not importable as a library — it installs a top-level package named
``src`` (v1.6) or none at all (v1.5), pins old numpy/scipy/pandas and writes to
``./result`` — so each version runs as a subprocess in a virtualenv of its own,
provisioned on first use under ``$OMNIDOCBENCH_EVAL_DIR`` (default
``~/.cache/olmo_eval/omnidocbench``). Set ``OMNIDOCBENCH_EVAL_REPO`` and
``OMNIDOCBENCH_EVAL_PYTHON`` to use an existing checkout and interpreter instead.

Evaluation is a dataset-level step: every metric is a mean over pages. One call to
:func:`run_official_evaluation` scores all pages and returns both the evaluator's own
leaderboard numbers and the per-page values, which the task records as each response's scorer
result (saved with the predictions).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer, get_scorer_result
from olmo_eval.common.types import Instance, LMOutput, Response

logger = logging.getLogger(__name__)

EVAL_REPO_URL = "https://github.com/opendatalab/OmniDocBench.git"

SCORER_NAME = "omnidocbench"

#: Binaries the CDM formula metric shells out to (LaTeX, ImageMagick 7, Ghostscript).
CDM_BINARIES = ("pdflatex", "magick", "gs")

_PRED_DIR_NAME = "pred"
_MATCH_METHOD = "quick_match"
_SAMPLE_KEY_RE = re.compile(r"^(?P<image>.*)_\[[^\]]*\]$")


#: CDM scores every render failure as zero without raising, so a missing package or font, a
#: TeX Live older than 2022 (no ``\mathcolor``) or an ImageMagick delegate Ghostscript cannot
#: serve would otherwise surface only as low formula scores. Each version's CDM template is
#: rendered once with a colored token and Chinese text before inference to rule that out.
_CDM_PROBE_BODY = r"\mathcolor[RGB]{255,0,0}{x}^{2} + \upalpha + \text{中文}"

#: v1.6's Chinese formula template: pdflatex with ``CJK`` and the ``gkai`` font.
_PDFLATEX_PROBE_TEX = (
    r"""
\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\geometry{a4paper,scale=0.98}
\pagestyle{empty}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{upgreek}
\usepackage{CJK}
\usepackage{amssymb}
\usepackage{xcolor}
\begin{document}
\begin{CJK}{UTF8}{gkai}
\begin{displaymath}
"""
    + _CDM_PROBE_BODY
    + r"""
\end{displaymath}
\end{CJK}
\end{document}
"""
)

#: v1.5's formula template: xelatex with ``xeCJK`` and the Source Han Sans SC font.
_XELATEX_PROBE_TEX = (
    r"""
\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\geometry{a4paper,scale=0.98}
\pagestyle{empty}
\usepackage{amsmath}
\usepackage{upgreek}
\usepackage{amssymb}
\usepackage{xcolor}
\usepackage{xeCJK}
\setCJKmainfont{Source Han Sans SC}
\setCJKsansfont{Source Han Sans SC}
\setCJKmonofont{Source Han Sans SC}
\xeCJKsetup{CJKmath=true}
\begin{document}
\begin{displaymath}
"""
    + _CDM_PROBE_BODY
    + r"""
\end{displaymath}
\end{document}
"""
)


@dataclass(frozen=True)
class EvaluatorVersion:
    """One release of the official evaluator and how to stand it up."""

    name: str
    repo_commit: str
    python_version: str
    #: ``uv pip install`` arguments; ``{repo}`` is the checkout.
    install: tuple[str, ...]
    #: Whether pages annotated with a table / display formula but left without a scored
    #: sample count as zero in the page average (v1.6 and later).
    zero_fills_missing_pages: bool
    #: Extra binaries CDM needs beyond :data:`CDM_BINARIES`.
    cdm_binaries: tuple[str, ...] = ()
    #: The LaTeX engine CDM renders formulas with, and a document in CDM's own formula
    #: template (with a colored token and Chinese text) that must render for CDM to work.
    cdm_latex: str = "pdflatex"
    cdm_probe_tex: str = ""
    #: Font families CDM's template names, as ``(family, archive URL, archive SHA-256)``.
    cdm_fonts: tuple[tuple[str, str, str], ...] = ()
    #: ``(installed, replacement)`` pairs swapped after installing, for dependencies that
    #: pull a variant the evaluation host cannot load.
    replacements: tuple[tuple[str, str], ...] = ()
    #: Modules the evaluator imports at startup, checked right after provisioning so a
    #: broken environment fails before inference rather than after it.
    startup_imports: tuple[str, ...] = ()


EVALUATORS: dict[str, EvaluatorVersion] = {
    # Scoring code of the v1.6 release (later commits on ``main`` touch only docs and tools).
    # The evaluator forks scoring workers from threads, which newer ``filelock`` releases
    # break; the failure is swallowed and the sample scored zero, hence the pin.
    "v1.6": EvaluatorVersion(
        name="v1.6",
        repo_commit="f133a71e9e91c3621c7ce8994200a7b394a06eb3",
        python_version="3.11",
        install=("-e", "{repo}", "filelock==3.16.1"),
        zero_fills_missing_pages=True,
        cdm_probe_tex=_PDFLATEX_PROBE_TEX,
        startup_imports=("src.cli",),
    ),
    # Head of the ``v1_5`` branch. Its ``requirements.txt`` pins a whole notebook
    # environment; these are the packages the evaluator imports, at those pins.
    "v1.5": EvaluatorVersion(
        name="v1.5",
        repo_commit="59b103c4b47d3a01fada83491585d6512a40c0bc",
        python_version="3.10",
        install=(
            "apted==1.0.3",
            "beautifulsoup4==4.11.1",
            "datasets==3.1.0",
            "evaluate==0.4.3",
            "filelock==3.16.1",
            "func-timeout==4.3.5",
            "Levenshtein==0.25.1",
            "loguru==0.7.2",
            "lxml==4.9.1",
            "matplotlib==3.7.5",
            "mmeval==0.2.1",
            "nltk==3.9.1",
            "numpy==1.24.4",
            "pandas==2.0.3",
            "pillow==10.4.0",
            "pycocotools==2.0.7",
            "pylatexenc==3.0a30",
            "PyYAML==6.0.2",
            "rapidfuzz==3.9.7",
            # CDM's RANSAC; imported lazily inside a bare ``except`` that turns a missing
            # package into a silent zero, and pinned by ``metrics/cdm/requirements.txt``.
            "scikit-image==0.20.0",
            "scipy==1.10.1",
            "tabulate==0.9.0",
            "tqdm==4.67.1",
        ),
        zero_fills_missing_pages=False,
        # v1.5 tokenizes formulas with the KaTeX parser through Node.js, and renders them with
        # xelatex in Source Han Sans SC (its CDM README).
        cdm_binaries=("node", "xelatex"),
        cdm_latex="xelatex",
        cdm_probe_tex=_XELATEX_PROBE_TEX,
        cdm_fonts=(
            (
                "Source Han Sans SC",
                "https://github.com/adobe-fonts/source-han-sans/releases/download/2.005R/"
                "09_SourceHanSansSC.zip",
                "ef7364f7ac2564be1ae9c1d74276de2653fe38b73449070398c4fc0b7e032ff1",
            ),
        ),
        # ``mmeval`` requires the GUI OpenCV build, which needs ``libGL`` at import time;
        # the headless build provides the same ``cv2`` without it.
        replacements=(("opencv-python", "opencv-python-headless==4.11.0.86"),),
        startup_imports=("dataset", "metrics", "task", "metrics.cdm_metric"),
    ),
}


def missing_cdm_binaries(version: str = "v1.6") -> list[str]:
    binaries = (*CDM_BINARIES, *EVALUATORS[version].cdm_binaries)
    return [binary for binary in binaries if not shutil.which(binary)]


# ---------------------------------------------------------------------------
# Evaluator provisioning
# ---------------------------------------------------------------------------


def _run(cmd: Sequence[str], **kwargs: Any) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(list(cmd), check=True, **kwargs)


def ensure_evaluator(version: str = "v1.6") -> tuple[Path, Path]:
    """Return ``(repo_dir, python)`` for the official evaluator, provisioning it if needed."""
    spec = EVALUATORS[version]
    repo_env, python_env = (
        os.environ.get("OMNIDOCBENCH_EVAL_REPO"),
        os.environ.get("OMNIDOCBENCH_EVAL_PYTHON"),
    )
    if repo_env and python_env:
        return Path(repo_env), Path(python_env)

    root = Path(
        os.environ.get("OMNIDOCBENCH_EVAL_DIR")
        or Path.home() / ".cache" / "olmo_eval" / "omnidocbench"
    )
    home = root / spec.repo_commit[:12]
    repo, venv, ready = home / "repo", home / "venv", home / ".ready"
    python = venv / "bin" / "python"
    if ready.exists():
        return repo, python

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "Provisioning the OmniDocBench evaluator needs `uv` on PATH (or point "
            "OMNIDOCBENCH_EVAL_REPO / OMNIDOCBENCH_EVAL_PYTHON at an existing install)."
        )
    from filelock import FileLock

    home.mkdir(parents=True, exist_ok=True)
    with FileLock(str(home / ".lock")):
        if ready.exists():
            return repo, python
        shutil.rmtree(repo, ignore_errors=True)
        shutil.rmtree(venv, ignore_errors=True)
        _run(["git", "clone", "--quiet", EVAL_REPO_URL, str(repo)])
        _run(["git", "-C", str(repo), "checkout", "--quiet", spec.repo_commit])
        _run([uv, "venv", "--quiet", "--python", spec.python_version, str(venv)])
        packages = [arg.format(repo=repo) for arg in spec.install]
        _run([uv, "pip", "install", "--quiet", "--python", str(python), *packages])
        for installed, replacement in spec.replacements:
            _run([uv, "pip", "uninstall", "--quiet", "--python", str(python), installed])
            _run([uv, "pip", "install", "--quiet", "--python", str(python), replacement])
        _check_startup_imports(spec, repo, python)
        ready.touch()
    return repo, python


def _check_startup_imports(spec: EvaluatorVersion, repo: Path, python: Path) -> None:
    if not spec.startup_imports:
        return
    script = "import importlib\n" + "".join(
        f"importlib.import_module({module!r})\n" for module in spec.startup_imports
    )
    proc = subprocess.run(
        [str(python), "-c", script], cwd=repo, capture_output=True, text=True, timeout=600
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-15:])
        raise RuntimeError(
            f"The OmniDocBench {spec.name} evaluator was installed but cannot start:\n{tail}"
        )


# ---------------------------------------------------------------------------
# CDM toolchain
# ---------------------------------------------------------------------------

#: What CDM's TeX templates load, on top of TeX Live's ``scheme-small``. ``was`` provides
#: ``upgreek``; ``cjk`` and ``arphic`` render v1.6's Chinese formulas, ``xecjk`` v1.5's.
_TEXLIVE_PACKAGES = (
    "cjk",
    "cjkutils",
    "arphic",
    "arphic-ttf",
    "was",
    "booktabs",
    "multirow",
    "xcolor",
    "geometry",
    "amsmath",
    "amsfonts",
    "standalone",
    "preview",
    "xecjk",
)
#: A style file from each package the templates load, to spot a TeX Live without them.
_TEXLIVE_STY_FILES = ("CJK.sty", "xeCJK.sty", "upgreek.sty", "booktabs.sty", "multirow.sty")
#: The CDM runtime OmniDocBench's README lists as verified (its Docker image): TeX Live 2025,
#: ImageMagick 7.1.1-47 and Ghostscript 9.55.0. Rendering differs between versions and CDM
#: follows it (Ghostscript 10.02 instead of 9.55 moved a formula score by 0.2 points), so these
#: exact versions are installed unless the ones on ``PATH`` already match. Every download is
#: checked against the SHA-256 recorded next to it. TeX Live's own packages are verified by
#: its installer against the repository's checksums.
_TEXLIVE_YEAR = "2025"
#: The frozen final TeX Live 2025 repository, from the historic archive and a mirror of it.
_TEXLIVE_REPOSITORIES = (
    "https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2025/tlnet-final",
    "https://mirrors.tuna.tsinghua.edu.cn/tex-historic-archive/systems/texlive/2025/tlnet-final",
)
_MAGICK_VERSION = "7.1.1-47"
_TEXLIVE_INSTALLER_SHA256 = "311df9f1477fd90c520159d1feddc2d6270f010d8349d1f6bdb9461a93b48a5c"
_MAGICK_APPIMAGE_URL = (
    "https://github.com/ImageMagick/ImageMagick/releases/download/7.1.1-47/"
    "ImageMagick-82572af-gcc-x86_64.AppImage"
)
_MAGICK_APPIMAGE_SHA256 = "dcdd2e135dd5f0701d992ccbbf744f51624441b5bc8eedcabe21516996e210ab"
#: System libraries the AppImage links against but does not bundle (Debian/Ubuntu names).
_MAGICK_SYSTEM_PACKAGES = (
    "libbrotli1",
    "libfontconfig1",
    "libfreetype6",
    "libfribidi0",
    "libharfbuzz0b",
    "libx11-6",
    "libxcb1",
    "libuuid1",
    "libexpat1",
    "libstdc++6",
)
_GHOSTSCRIPT_VERSION = "9.55.0"
_GHOSTSCRIPT_URL = (
    "https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs9550/"
    "ghostscript-9.55.0-linux-x86_64.tgz"
)
_GHOSTSCRIPT_SHA256 = "e8756b70bd584ef1982577f88223aac95126065b2906a5b2a37931147ac6ab00"
#: The Node.js release v1.5's CDM README installs (its SHA-256 as nodejs.org publishes it).
_NODE_VERSION = "v16.13.1"
_NODE_SHA256 = "a3721f87cecc0b52b0be8587c20776ac7305db413751db02c55aa2bffac15198"

_cdm_ready: set[str] = set()


def ensure_cdm_toolchain(version: str = "v1.6") -> None:
    """Put CDM's pinned toolchain on ``PATH``, installing what is missing, and check it renders.

    Tools that are missing or at another version are installed under ``$OMNIDOCBENCH_CDM_DIR``
    (default ``~/.cache/olmo_eval/omnidocbench/cdm``), which goes first on ``PATH``: TeX Live
    2025 ``scheme-small`` with :data:`_TEXLIVE_PACKAGES`, the ImageMagick 7.1.1-47 AppImage
    (plus, from apt, the system libraries it links against), Ghostscript 9.55.0, and Node.js
    for versions that tokenize with KaTeX.
    """
    if version in _cdm_ready:
        return
    root = Path(
        os.environ.get("OMNIDOCBENCH_CDM_DIR")
        or Path.home() / ".cache" / "olmo_eval" / "omnidocbench" / "cdm"
    )
    _add_cdm_dirs_to_path(root)
    if tools := _cdm_tools_to_install(version):
        import platform
        import sys

        if not sys.platform.startswith("linux") or platform.machine() not in ("x86_64", "AMD64"):
            raise RuntimeError(
                f"CDM needs {', '.join(tools)} at the pinned versions, and they can only be "
                f"installed automatically on Linux x86_64 (this host is {sys.platform} "
                f"{platform.machine()}). Put them on PATH first."
            )
        from filelock import FileLock

        root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(root / ".lock")):
            tools = _cdm_tools_to_install(version)
            packages = []
            if "magick" in tools and shutil.which("apt-get"):
                packages.extend(_MAGICK_SYSTEM_PACKAGES)
            if "pdflatex" in tools and shutil.which("perl") is None:
                packages.append("perl")
            if "fonts" in tools and shutil.which("fc-cache") is None:
                packages.append("fontconfig")
            if packages:
                _apt_install(*packages)
            if "gs" in tools:
                _install_ghostscript(root)
            if "magick" in tools:
                _install_magick(root)
            if "pdflatex" in tools:
                _install_texlive(root)
            elif "texlive-packages" in tools:
                _install_texlive_packages(root)
            if "node" in tools:
                _install_node(root)
            if "fonts" in tools:
                _install_fonts(root, EVALUATORS[version].cdm_fonts)
        _add_cdm_dirs_to_path(root)
        tools = _cdm_tools_to_install(version)
        if tools:
            raise RuntimeError(f"Could not install CDM's pinned {', '.join(tools)}.")
    _probe_cdm_render(EVALUATORS[version])
    _cdm_ready.add(version)


def _cdm_tools_to_install(version: str) -> list[str]:
    """CDM tools missing from ``PATH`` or at a version other than the pinned one."""
    pinned = {
        "gs": (["gs", "--version"], _GHOSTSCRIPT_VERSION),
        "magick": (["magick", "--version"], f"ImageMagick {_MAGICK_VERSION} "),
        "pdflatex": (["pdflatex", "--version"], f"(TeX Live {_TEXLIVE_YEAR})"),
    }
    tools = [tool for tool, (cmd, expected) in pinned.items() if expected not in _run_quiet(cmd)]
    if "pdflatex" not in tools:
        found = _run_quiet(["kpsewhich", *_TEXLIVE_STY_FILES]).split()
        if len(found) < len(_TEXLIVE_STY_FILES):
            tools.append("texlive-packages")
    tools += [b for b in EVALUATORS[version].cdm_binaries if not shutil.which(b)]
    families = _run_quiet(["fc-list", ":", "family"])
    if any(family not in families for family, *_ in EVALUATORS[version].cdm_fonts):
        tools.append("fonts")
    return tools


def _run_quiet(cmd: Sequence[str]) -> str:
    if not shutil.which(cmd[0]):
        return ""
    try:
        return subprocess.run(list(cmd), capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _add_cdm_dirs_to_path(root: Path) -> None:
    current = os.environ.get("PATH", "").split(os.pathsep)
    dirs = [root / "texlive" / "bin" / "x86_64-linux", root / "bin", root / "node" / "bin"]
    new = [str(d) for d in dirs if d.is_dir() and str(d) not in current]
    if new:
        os.environ["PATH"] = os.pathsep.join([*new, *current])


def _download(url: str, dest: Path, sha256: str) -> Path:
    """Download ``url`` to ``dest`` and check it against ``sha256`` before anything runs it."""
    import hashlib
    import ssl
    import urllib.request

    # Interpreters built outside the system (uv, conda) may not find the system CA store.
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        logger.info("Downloading %s", url)
        try:
            with (
                urllib.request.urlopen(url, context=context, timeout=300) as response,
                open(dest, "wb") as out,
            ):
                shutil.copyfileobj(response, out)
            break
        except OSError:
            if attempt == 2:
                raise
            logger.warning("Download of %s failed; retrying.", url, exc_info=True)
    digest = hashlib.sha256()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        dest.unlink()
        raise RuntimeError(f"{url} has SHA-256 {digest.hexdigest()}, expected {sha256}.")
    return dest


def _apt_install(*packages: str) -> None:
    if shutil.which("apt-get") is None:
        raise RuntimeError(f"CDM needs {', '.join(packages)}, and apt-get is not available.")
    env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
    _run(["apt-get", "update", "-qq"], env=env)
    _run(["apt-get", "install", "-y", "-qq", "--no-install-recommends", *packages], env=env)


def _install_magick(root: Path) -> None:
    appimage = _download(_MAGICK_APPIMAGE_URL, root / "magick.AppImage", _MAGICK_APPIMAGE_SHA256)
    appimage.chmod(0o755)
    shutil.rmtree(root / "squashfs-root", ignore_errors=True)
    # Extracting avoids mounting the AppImage, which needs FUSE.
    _run([str(appimage), "--appimage-extract"], cwd=root, stdout=subprocess.DEVNULL)
    delegates = root / "squashfs-root" / "usr" / "etc" / "ImageMagick-7" / "delegates.xml"
    # ImageMagick's configure writes the alpha device the local Ghostscript supports into
    # the PDF delegate. The AppImage was built against one with ``png16malpha``; Ghostscript
    # 9.55 lacks it, and a source build against 9.55 (the verified runtime) uses ``pngalpha``.
    delegates.write_text(delegates.read_text().replace("png16malpha", "pngalpha"))
    (root / "bin").mkdir(exist_ok=True)
    link = root / "bin" / "magick"
    link.unlink(missing_ok=True)
    link.symlink_to(root / "squashfs-root" / "AppRun")


def _install_texlive(root: Path) -> None:
    for repository in _TEXLIVE_REPOSITORIES:
        try:
            _install_texlive_from(root, repository)
            return
        except (OSError, subprocess.CalledProcessError, StopIteration):
            if repository == _TEXLIVE_REPOSITORIES[-1]:
                raise
            logger.warning("TeX Live install from %s failed; trying the next.", repository)


def _install_texlive_from(root: Path, repository: str) -> None:
    import tarfile

    work = root / "install-tl"
    texdir = root / "texlive"
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(texdir, ignore_errors=True)
    archive = _download(
        f"{repository}/install-tl-unx.tar.gz",
        work / "install-tl-unx.tar.gz",
        _TEXLIVE_INSTALLER_SHA256,
    )
    with tarfile.open(archive) as tar:
        tar.extractall(work, filter="data")
    installer = next(work.glob("install-tl-*/install-tl"))
    profile = work / "texlive.profile"
    profile.write_text(
        "selected_scheme scheme-small\n"
        f"TEXDIR {texdir}\n"
        f"TEXMFLOCAL {texdir}/texmf-local\n"
        f"TEXMFSYSCONFIG {texdir}/texmf-config\n"
        f"TEXMFSYSVAR {texdir}/texmf-var\n"
        f"TEXMFHOME {texdir}/texmf-home\n"
        f"TEXMFCONFIG {texdir}/texmf-config-user\n"
        f"TEXMFVAR {texdir}/texmf-var-user\n"
        "instopt_adjustpath 0\n"
        "tlpdbopt_autobackup 0\n"
        "tlpdbopt_install_docfiles 0\n"
        "tlpdbopt_install_srcfiles 0\n"
    )
    _run(
        ["perl", str(installer), "--profile", str(profile), "--repository", repository],
        cwd=installer.parent,
    )
    _run([str(texdir / "bin" / "x86_64-linux" / "tlmgr"), "install", *_TEXLIVE_PACKAGES])
    shutil.rmtree(work, ignore_errors=True)


def _install_texlive_packages(root: Path) -> None:
    tlmgr = root / "texlive" / "bin" / "x86_64-linux" / "tlmgr"
    if not tlmgr.exists():
        # A TeX Live 2025 installed elsewhere lacks packages CDM needs; install our own.
        _install_texlive(root)
        return
    _run([str(tlmgr), "install", *_TEXLIVE_PACKAGES])


def _install_ghostscript(root: Path) -> None:
    import tarfile

    archive = _download(_GHOSTSCRIPT_URL, root / "ghostscript.tgz", _GHOSTSCRIPT_SHA256)
    (root / "bin").mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        member = next(
            m for m in tar.getmembers() if m.name.endswith("-linux-x86_64") and m.isfile()
        )
        member.name = "gs"
        tar.extract(member, root / "bin", filter="data")
    (root / "bin" / "gs").chmod(0o755)
    archive.unlink()


def _install_fonts(root: Path, fonts: Sequence[tuple[str, str, str]]) -> None:
    import zipfile

    # A per-user font directory fontconfig reads by default, so xelatex finds the families.
    target = Path.home() / ".local" / "share" / "fonts" / "omnidocbench"
    target.mkdir(parents=True, exist_ok=True)
    for _, url, sha256 in fonts:
        archive = _download(url, root / Path(url).name, sha256)
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if name.lower().endswith((".otf", ".ttf", ".ttc")):
                    (target / Path(name).name).write_bytes(zf.read(name))
        archive.unlink()
    if shutil.which("fc-cache"):
        _run(["fc-cache", "-f", str(target)])


def _install_node(root: Path) -> None:
    import tarfile

    name = f"node-{_NODE_VERSION}-linux-x64"
    archive = _download(
        f"https://nodejs.org/dist/{_NODE_VERSION}/{name}.tar.xz",
        root / f"{name}.tar.xz",
        _NODE_SHA256,
    )
    target = root / "node"
    shutil.rmtree(target, ignore_errors=True)
    with tarfile.open(archive) as tar:
        members = []
        for member in tar.getmembers():
            member.name = member.name.partition("/")[2]
            if member.name:
                members.append(member)
        tar.extractall(target, members=members, filter="data")
    archive.unlink()


def _probe_cdm_render(spec: EvaluatorVersion) -> None:
    with tempfile.TemporaryDirectory(prefix="omnidocbench_cdm_probe_") as tmp:
        work = Path(tmp)
        (work / "probe.tex").write_text(spec.cdm_probe_tex, encoding="utf-8")
        steps = (
            [spec.cdm_latex, "-interaction=nonstopmode", "-halt-on-error", "probe.tex"],
            ["magick", "-density", "200", "-quality", "100", "probe.pdf", "probe.png"],
        )
        for cmd in steps:
            proc = subprocess.run(
                cmd, cwd=work, capture_output=True, text=True, errors="replace", timeout=300
            )
            if proc.returncode != 0:
                tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-20:])
                raise RuntimeError(
                    f"CDM's toolchain cannot render a {spec.name} formula ({cmd[0]} exited "
                    f"{proc.returncode}); formulas would all score zero.\n{tail}"
                )
        if not (work / "probe.png").exists():
            raise RuntimeError(f"CDM's toolchain rendered no image for a {spec.name} formula.")


# ---------------------------------------------------------------------------
# Running the evaluator
# ---------------------------------------------------------------------------


@dataclass
class OfficialEvaluation:
    """Outcome of one official end-to-end evaluation."""

    #: ``image_name -> {text_edit, read_order_edit, formula_edit, table_teds,
    #: table_teds_s, formula_cdm}``; a key is absent when the page has no such sample.
    per_page: dict[str, dict[str, float]] = field(default_factory=dict)
    #: The evaluator's own page-averaged numbers, in its raw 0-1 units, under the same keys
    #: (plus ``overall`` on the leaderboard's 0-100 scale when CDM ran).
    summary: dict[str, float] = field(default_factory=dict)
    #: Samples the evaluator scored zero because its scoring code crashed or timed out.
    num_scorer_failures: int = 0
    #: Worker threads the evaluator ran with. v1.6 zeroes pages whose matching passes a time
    #: limit, so the zeroed count, and the scores, can move with workers and host load.
    workers: int = 0


def _config(
    version: str, gt_path: Path, pred_dir: Path, *, with_cdm: bool, workers: int
) -> dict[str, Any]:
    formula_metrics = ["Edit_dist", "CDM"] if with_cdm else ["Edit_dist"]
    metrics: dict[str, dict[str, Any]] = {
        "text_block": {"metric": ["Edit_dist"]},
        "display_formula": {"metric": formula_metrics},
        "table": {"metric": ["TEDS", "Edit_dist"]},
        "reading_order": {"metric": ["Edit_dist"]},
    }
    dataset: dict[str, Any] = {
        "dataset_name": "end2end_dataset",
        "ground_truth": {"data_path": str(gt_path)},
        "prediction": {"data_path": str(pred_dir)},
        "match_method": _MATCH_METHOD,
    }
    if version == "v1.6":
        metrics["display_formula"]["cdm_workers"] = workers
        metrics["table"]["teds_workers"] = workers
        dataset.update(
            {
                "match_workers": workers,
                "quick_match_truncated_timeout_sec": 300,
                "match_timeout_sec": 420,
                "timeout_fallback_max_chunk_span": 10,
                "timeout_fallback_order_penalty": 0.10,
            }
        )
    return {"end2end_eval": {"metrics": metrics, "dataset": dataset}}


def _load(result_dir: Path, name: str) -> Any:
    path = result_dir / f"{_PRED_DIR_NAME}_{_MATCH_METHOD}_{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _page_means(per_sample: Mapping[str, Any] | None, field_name: str | None) -> dict[str, float]:
    """Mean per page of a ``"<image>_[gt idx]" -> score`` map."""
    by_page: dict[str, list[float]] = {}
    for key, value in (per_sample or {}).items():
        match = _SAMPLE_KEY_RE.match(key)
        score = value[field_name] if field_name is not None else value
        if match is not None and isinstance(score, (int, float)):
            by_page.setdefault(match.group("image"), []).append(float(score))
    return {image: sum(scores) / len(scores) for image, scores in by_page.items()}


def _summary(metric_result: Mapping[str, Any]) -> dict[str, float]:
    """The leaderboard numbers as ``tools/generate_result_tables.ipynb`` reads them."""

    def page_avg(element: str, metric: str) -> float | None:
        value = ((metric_result.get(element) or {}).get("page") or {}).get(metric)
        return float(value["ALL"]) if value and "ALL" in value else None

    def edit(element: str) -> float | None:
        value = ((metric_result.get(element) or {}).get("all") or {}).get("Edit_dist") or {}
        return float(value["ALL_page_avg"]) if "ALL_page_avg" in value else None

    values = {
        "text_edit": edit("text_block"),
        "read_order_edit": edit("reading_order"),
        "table_teds": page_avg("table", "TEDS"),
        "table_teds_s": page_avg("table", "TEDS_structure_only"),
        "formula_cdm": page_avg("display_formula", "CDM"),
    }
    summary = {k: v for k, v in values.items() if v is not None}
    if all(k in summary for k in ("text_edit", "table_teds", "formula_cdm")):
        summary["overall"] = (
            (1.0 - summary["text_edit"]) * 100
            + summary["table_teds"] * 100
            + summary["formula_cdm"] * 100
        ) / 3.0
    return summary


def _count_failures(metric_result: Mapping[str, Any], log_text: str) -> int:
    count = 0
    for element in ("table", "display_formula"):
        for debug in ((metric_result.get(element) or {}).get("metric_debug") or {}).values():
            for counter in ("error_case_count", "timeout_case_count", "exception_case_count"):
                count += int(debug.get(counter) or 0)
    fallbacks = (metric_result.get("match_debug") or {}).get("text_match_fallback_counts") or {}
    count += sum(int(v or 0) for v in fallbacks.values())
    if count == 0:
        # Older evaluators only report a zeroed sample on stdout.
        count = len(re.findall(r"score is set to 0", log_text))
    return count


#: Runs ``pdf_validation.py`` with a deeper recursion limit. The v1.6 display-formula
#: matcher recurses once per predicted formula on a page, so a page with a thousand formulas
#: (a model stuck repeating one) crashes the whole evaluation at Python's default limit of
#: 1,000. Its candidate search is capped, so a deeper limit changes no score. Matching runs in
#: worker threads, which get a larger stack to match.
_LAUNCH_SCRIPT = """
import runpy, sys, threading
sys.setrecursionlimit(100_000)
threading.stack_size(512 * 1024 * 1024)
repo, script = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
sys.argv = [script, *sys.argv[3:]]
runpy.run_path(script, run_name="__main__")
"""

#: Prepended for v1.6, whose matching and TEDS worker threads start a helper process per
#: LaTeX conversion and table, and abandon threads that pass the page timeout. Forking from
#: those threads can leave a lock held forever (loguru takes its locks around every fork), which
#: hangs the evaluation with no output. A fork server starts the helpers from a single-threaded
#: process instead; they compute the same values. The preload spares each helper a fresh import
#: of the evaluator.
_FORKSERVER_PRELUDE = """
import multiprocessing
multiprocessing.set_start_method("forkserver")
multiprocessing.set_forkserver_preload(
    ["src.cli", "src.core.preprocess.text_postprocess", "src.metrics.cal_metric"]
)
"""

#: Wall-clock cap on one evaluation, overridable with ``OMNIDOCBENCH_EVAL_TIMEOUT_S``. A hung
#: evaluator otherwise holds the run until the job's own limit and loses every prediction with
#: it; past the cap the evaluation is reported as failed and the predictions are kept.
_DEFAULT_EVAL_TIMEOUT_S = 12 * 3600


def run_official_evaluation(
    predictions: Mapping[str, str],
    gt_pages: Sequence[Mapping[str, Any]],
    *,
    with_cdm: bool,
    version: str = "v1.6",
    workers: int | None = None,
) -> OfficialEvaluation:
    """Score ``predictions`` (``image_name -> markdown``) against their annotated pages."""
    import yaml

    repo, python = ensure_evaluator(version)
    if with_cdm:
        ensure_cdm_toolchain(version)
    if workers is None:
        workers = int(os.environ.get("OMNIDOCBENCH_EVAL_WORKERS") or 0) or max(
            1, min(16, (os.cpu_count() or 4) // 4)
        )

    with tempfile.TemporaryDirectory(prefix="omnidocbench_eval_") as tmp:
        work = Path(tmp)
        pred_dir = work / _PRED_DIR_NAME
        pred_dir.mkdir()
        for image_name, markdown in predictions.items():
            # The evaluator's first lookup for image "X.ext" is "X.md".
            (pred_dir / f"{image_name[:-4]}.md").write_text(markdown, encoding="utf-8")
        gt_path = work / "gt.json"
        gt_path.write_text(json.dumps(list(gt_pages), ensure_ascii=False), encoding="utf-8")
        config_path = work / "end2end.yaml"
        config_path.write_text(
            yaml.safe_dump(_config(version, gt_path, pred_dir, with_cdm=with_cdm, workers=workers))
        )
        result_dir = work / "result"
        result_dir.mkdir()

        log_path = work / "evaluator.log"
        timeout_s = float(os.environ.get("OMNIDOCBENCH_EVAL_TIMEOUT_S") or _DEFAULT_EVAL_TIMEOUT_S)
        with open(log_path, "w") as log:
            try:
                proc = subprocess.run(
                    [
                        str(python),
                        "-c",
                        (_FORKSERVER_PRELUDE if version == "v1.6" else "") + _LAUNCH_SCRIPT,
                        str(repo),
                        str(repo / "pdf_validation.py"),
                        "--config",
                        str(config_path),
                    ],
                    cwd=work,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                )
                failure = f"failed (exit {proc.returncode})" if proc.returncode else None
            except subprocess.TimeoutExpired:
                failure = f"did not finish within {timeout_s:.0f}s"
        log_text = log_path.read_text(errors="replace")
        metric_result = None if failure else _load(result_dir, "metric_result")
        if not metric_result:
            tail = "\n".join(log_text.splitlines()[-40:])
            raise RuntimeError(
                f"OmniDocBench {version} evaluator {failure or 'wrote no metrics'}. "
                f"Log tail:\n{tail}"
            )

        per_page: dict[str, dict[str, float]] = {}
        page_values = {
            "text_edit": _load(result_dir, "text_block_per_page_edit") or {},
            "read_order_edit": _load(result_dir, "reading_order_per_page_edit") or {},
            "formula_edit": _load(result_dir, "display_formula_per_page_edit") or {},
            "table_teds": _page_means(_load(result_dir, "table_per_table_TEDS"), "TEDS"),
            "table_teds_s": _page_means(
                _load(result_dir, "table_per_table_TEDS"), "TEDS_structure_only"
            ),
            "formula_cdm": _page_means(_load(result_dir, "display_formula_per_sample_CDM"), None),
        }
        for name, values in page_values.items():
            for image_name, value in values.items():
                if isinstance(value, (int, float)):
                    per_page.setdefault(image_name, {})[name] = float(value)

        failures = _count_failures(metric_result, log_text)
        if failures:
            logger.error(
                "OmniDocBench evaluator scored %d sample(s) zero after an internal crash, "
                "timeout or matching fallback; see n_scorer_failures.",
                failures,
            )
        return OfficialEvaluation(
            per_page=per_page,
            summary=_summary(metric_result),
            num_scorer_failures=failures,
            workers=workers,
        )


# ---------------------------------------------------------------------------
# Scorer + metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OmniDocBenchScorer(Scorer):
    """Names the OmniDocBench score channel.

    Pages are graded together by :func:`run_official_evaluation`, and the task records each
    page's values as this scorer's result; the per-response score is the page's text accuracy
    (``1 - text edit distance``).
    """

    name: str = SCORER_NAME

    def score(self, instance: Instance, output: LMOutput) -> float:
        result = get_scorer_result(output, self.name) or {}
        text_edit = result.get("text_edit")
        return 1.0 - text_edit if text_edit is not None else 0.0


def _page_result(response: Response) -> dict[str, Any] | None:
    return get_scorer_result(response.outputs[0], SCORER_NAME) if response.outputs else None


def _page_results(responses: Sequence[Response]) -> Iterator[tuple[Response, dict[str, Any]]]:
    for response in responses:
        result = _page_result(response)
        if result is not None:
            yield response, result


@dataclass(frozen=True)
class OmniDocBenchPageMetric(Metric):
    """Page average of one evaluator component, optionally within a page-attribute slice.

    ``zero_fill`` names the instance-metadata flag marking pages annotated with the element
    (``has_table`` / ``has_formula``): such a page counts as zero when no sample was scored
    for it, as the v1.6 evaluator does. Without it, only pages that have a value are
    averaged. ``scale`` converts to the leaderboard's unit (x100 for TEDS and CDM).
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    component: str = ""
    zero_fill: str | None = None
    scale: float = 1.0
    attribute: str | None = None
    attribute_value: str | None = None

    def _page_value(self, response: Response, result: dict[str, Any]) -> float | None:
        """The page's raw value, or ``None`` when the page is outside the metric's scope."""
        meta = response.instance.metadata
        if self.attribute is not None and meta.get(self.attribute) != self.attribute_value:
            return None
        value = result.get(self.component)
        if value is None and self.zero_fill is not None and meta.get(self.zero_fill):
            value = 0.0
        return float(value) if value is not None else None

    def values(self, responses: Sequence[Response]) -> list[float]:
        vals = (self._page_value(response, result) for response, result in _page_results(responses))
        return [v for v in vals if v is not None]

    def compute(self, responses: Sequence[Response]) -> float:
        vals = self.values(responses)
        return sum(vals) / len(vals) * self.scale if vals else 0.0

    def compute_instance(self, response: Response) -> float | None:
        result = _page_result(response)
        value = self._page_value(response, result) if result is not None else None
        return value * self.scale if value is not None else None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False

    def pairwise_higher_is_better(self) -> bool:
        # Edit distances (text, formula, reading order) are better when lower.
        return not self.component.endswith("_edit")


@dataclass(frozen=True)
class OmniDocBenchOverallMetric(Metric):
    """The leaderboard Overall: ``((1 - TextEdit) * 100 + TableTEDS + FormulaCDM) / 3``."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    #: Whether missing table / formula pages count as zero (see the page metric).
    zero_fill: bool = True

    def _components(self) -> tuple[OmniDocBenchPageMetric, ...]:
        return (
            OmniDocBenchPageMetric("", self.scorer, "text_edit"),
            OmniDocBenchPageMetric(
                "",
                self.scorer,
                "table_teds",
                zero_fill="has_table" if self.zero_fill else None,
                scale=100.0,
            ),
            OmniDocBenchPageMetric(
                "",
                self.scorer,
                "formula_cdm",
                zero_fill="has_formula" if self.zero_fill else None,
                scale=100.0,
            ),
        )

    def compute(self, responses: Sequence[Response]) -> float:
        text, teds, cdm = (metric.compute(responses) for metric in self._components())
        return ((1.0 - text) * 100.0 + teds + cdm) / 3.0

    def compute_instance(self, response: Response) -> float | None:
        """The same terms on one page, averaged over those the page has (0-100)."""
        text, teds, cdm = (metric.compute_instance(response) for metric in self._components())
        if text is None:
            return None
        terms = [(1.0 - text) * 100.0, *(v for v in (teds, cdm) if v is not None)]
        return sum(terms) / len(terms)

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False
