"""Turn submitted deliverables into the text and images a judge reads.

Runs inside the AA-Briefcase-Lite sandbox image, which has LibreOffice, FFmpeg
and the Python document stack:

    python parse_submission.py SUBMISSION_DIR OUTPUT_DIR

Every file in SUBMISSION_DIR becomes a list of blocks in OUTPUT_DIR/manifest.json,
keyed by file name. A block is ``{"type": "text", "text": ...}``,
``{"type": "image", "path": ...}`` with the path of a JPEG relative to OUTPUT_DIR,
or ``{"type": "note", "text": ...}`` for what could not be parsed or was left out.
A file the parser cannot read gets a note; a tool that times out fails the run.

Documents are given both as text and as page images, the way Artificial
Analysis parses submissions for its judges: PDFs directly, PowerPoint and Excel
files through a LibreOffice PDF rendering. Spreadsheet cells carry their
formulas and their values as LibreOffice recalculates them. A video is given as
its stream metadata and frames sampled evenly through it; its audio is not
transcribed.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

#: Largest image handed to the judge, in pixels.
MAX_IMAGE_PIXELS = 1_000_000
#: JPEG quality of page images and video frames, which keeps a request with dozens
#: of them well under the judge API's size limit.
JPEG_QUALITY = 85
#: Characters of text kept per file.
MAX_TEXT_CHARS = 200_000
#: Page images kept per document type.
MAX_PAGES = {".pdf": 20, ".pptx": 40, ".xlsx": 10}
#: Cells listed per worksheet.
MAX_CELLS_PER_SHEET = 4_000
VIDEO_FRAMES = 16
TEXT_SUFFIXES = {".tex", ".srt", ".txt", ".md", ".csv", ".json", ".html", ".vtt"}
COMMAND_TIMEOUT = 600

#: LibreOffice settings that recalculate every formula when a workbook is loaded.
#: Its default trusts the values cached in the file, which some writers leave at 0.
RECALCULATE_ON_LOAD = """\
<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry"
    xmlns:xs="http://www.w3.org/2001/XMLSchema"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<item oor:path="/org.openoffice.Office.Calc/Formula/Load">
<prop oor:name="OOXMLRecalcMode" oor:op="fuse"><value>0</value></prop></item>
<item oor:path="/org.openoffice.Office.Calc/Formula/Load">
<prop oor:name="ODFRecalcMode" oor:op="fuse"><value>0</value></prop></item>
</oor:items>
"""


def text_block(text: str) -> dict:
    if len(text) > MAX_TEXT_CHARS:
        omitted = len(text) - MAX_TEXT_CHARS
        text = text[:MAX_TEXT_CHARS] + f"\n[... {omitted} more characters not shown ...]"
    return {"type": "text", "text": text}


def note(text: str) -> dict:
    return {"type": "note", "text": text}


def scale_for(width: float, height: float, max_pixels: int = MAX_IMAGE_PIXELS) -> float:
    """Zoom that fits a width by height page into ``max_pixels``, never enlarging past 2x."""
    return min(2.0, math.sqrt(max_pixels / (width * height)))


def frame_times(duration: float, count: int = VIDEO_FRAMES) -> list[float]:
    """Times at the middle of ``count`` equal slices of a video."""
    if duration <= 0:
        return [0.0]
    return [duration * (i + 0.5) / count for i in range(count)]


def render_pdf(pdf: Path, out_dir: Path, prefix: str, max_pages: int) -> tuple[list[str], int]:
    """Render up to ``max_pages`` pages to JPEG and return their paths and the page count."""
    import pymupdf

    paths: list[str] = []
    with pymupdf.open(pdf) as doc:
        for index, page in enumerate(doc):
            if index >= max_pages:
                break
            zoom = scale_for(page.rect.width, page.rect.height)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            path = out_dir / f"{prefix}-page-{index + 1:03d}.jpg"
            pixmap.save(path, jpg_quality=JPEG_QUALITY)
            paths.append(path.name)
        return paths, doc.page_count


def page_images(pdf: Path, out_dir: Path, prefix: str, max_pages: int) -> list[dict]:
    paths, pages = render_pdf(pdf, out_dir, prefix, max_pages)
    blocks: list[dict] = [{"type": "image", "path": path} for path in paths]
    if pages > max_pages:
        blocks.append(note(f"Only the first {max_pages} of {pages} pages are shown as images."))
    return blocks


def convert_with_libreoffice(path: Path, target: str, work_dir: Path) -> Path:
    """Convert a file with headless LibreOffice and return the converted file."""
    profile = work_dir / "libreoffice-profile"
    settings = profile / "user" / "registrymodifications.xcu"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(RECALCULATE_ON_LOAD)
    subprocess.run(
        ["soffice", f"-env:UserInstallation={profile.as_uri()}", "--headless"]
        + ["--convert-to", target, "--outdir", str(work_dir), str(path)],
        check=True,
        capture_output=True,
        timeout=COMMAND_TIMEOUT,
    )
    converted = work_dir / f"{path.stem}.{target.split(':')[0]}"
    if not converted.is_file():
        raise RuntimeError(f"LibreOffice did not write {converted.name}")
    return converted


def parse_pdf(path: Path, out_dir: Path) -> list[dict]:
    import pymupdf

    with pymupdf.open(path) as doc:
        pages = [f"--- Page {i + 1} ---\n{page.get_text()}" for i, page in enumerate(doc)]
    return [text_block("\n".join(pages)), *page_images(path, out_dir, path.name, MAX_PAGES[".pdf"])]


def parse_pptx(path: Path, out_dir: Path, work_dir: Path) -> list[dict]:
    from pptx import Presentation

    slides: list[str] = []
    for number, slide in enumerate(Presentation(str(path)).slides, start=1):
        lines = [f"--- Slide {number} ---"]
        for shape in slide.shapes:
            lines.extend(_shape_text(shape))
        notes_frame = slide.notes_slide.notes_text_frame if slide.has_notes_slide else None
        if notes_frame is not None and notes_frame.text.strip():
            lines.append(f"[Speaker notes] {notes_frame.text.strip()}")
        slides.append("\n".join(lines))
    blocks = [text_block("\n\n".join(slides))]
    try:
        pdf = convert_with_libreoffice(path, "pdf", work_dir)
        blocks.extend(page_images(pdf, out_dir, path.name, MAX_PAGES[".pptx"]))
    except Exception as e:
        blocks.append(note(f"The slides could not be rendered as images: {e}"))
    return blocks


def _shape_text(shape) -> list[str]:
    lines: list[str] = []
    if getattr(shape, "shape_type", None) == 6:  # a group: read its members
        for member in shape.shapes:
            lines.extend(_shape_text(member))
        return lines
    if getattr(shape, "has_text_frame", False) and shape.text_frame.text.strip():
        lines.append(shape.text_frame.text.strip())
    if getattr(shape, "has_table", False):
        for row in shape.table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))
    if getattr(shape, "has_chart", False):
        lines.extend(_chart_text(shape.chart))
    return lines


def _chart_text(chart) -> list[str]:
    """A chart's title and series values, or a bare marker if its data cannot be read."""
    try:
        title = chart.chart_title.text_frame.text if chart.has_title else ""
        lines = [f"[Chart: {title}]" if title else "[Chart]"]
        for plot in chart.plots:
            categories = [str(c) for c in plot.categories]
            for series in plot.series:
                values = ", ".join(
                    f"{c}={v}" for c, v in zip(categories, series.values, strict=False)
                )
                lines.append(f"  {series.name}: {values}")
        return lines
    except Exception:
        return ["[Chart whose data could not be read]"]


def parse_xlsx(path: Path, out_dir: Path, work_dir: Path) -> list[dict]:
    import openpyxl

    blocks: list[dict] = []
    try:
        recalculated = convert_with_libreoffice(path, "xlsx:Calc MS Excel 2007 XML", work_dir)
    except Exception as e:
        recalculated = path
        blocks.append(note(f"Formulas could not be recalculated: {e}"))
    formulas = openpyxl.load_workbook(path, data_only=False)
    values = openpyxl.load_workbook(recalculated, data_only=True)
    sheets: list[str] = []
    for sheet in formulas.worksheets:
        value_sheet = values[sheet.title] if sheet.title in values.sheetnames else None
        lines = [f"--- Sheet: {sheet.title} ({sheet.dimensions}) ---"]
        listed = 0
        for row in sheet.iter_rows():
            for cell in row:
                if cell.value is None:
                    continue
                if listed == MAX_CELLS_PER_SHEET:
                    lines.append(f"[... cells after {cell.coordinate} not listed ...]")
                    break
                lines.append(_cell_line(cell, value_sheet))
                listed += 1
            else:
                continue
            break
        sheets.append("\n".join(lines))
    blocks.insert(0, text_block("\n\n".join(sheets)))
    try:
        pdf = convert_with_libreoffice(path, "pdf", work_dir)
        blocks.extend(page_images(pdf, out_dir, path.name, MAX_PAGES[".xlsx"]))
    except Exception as e:
        blocks.append(note(f"The workbook could not be rendered as images: {e}"))
    return blocks


def _cell_line(cell, value_sheet) -> str:
    # An array formula is an object holding its formula text.
    formula = getattr(cell.value, "text", cell.value)
    if isinstance(formula, str) and formula.startswith("="):
        value = value_sheet[cell.coordinate].value if value_sheet is not None else None
        return f"{cell.coordinate}: {formula} -> {value}"
    return f"{cell.coordinate}: {cell.value}"


def parse_video(path: Path, out_dir: Path) -> list[dict]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams"]
        + [str(path)],
        check=True,
        capture_output=True,
        timeout=COMMAND_TIMEOUT,
    )
    info = json.loads(probe.stdout)
    duration = float(info.get("format", {}).get("duration") or 0.0)
    streams = []
    for stream in info.get("streams", []):
        kind = stream.get("codec_type")
        detail = f"{kind} ({stream.get('codec_name')}"
        if kind == "video":
            detail += f", {stream.get('width')}x{stream.get('height')}"
        streams.append(detail + ")")
    blocks = [
        text_block(f"Video duration: {duration:.1f} s. Streams: {', '.join(streams) or 'none'}."),
        note(
            f"The judge sees {VIDEO_FRAMES} frames sampled evenly through the video; "
            "its audio is not transcribed."
        ),
    ]
    # Shrink frames over the pixel budget, keeping the aspect ratio and even dimensions.
    fit = f"trunc(iw*sqrt({MAX_IMAGE_PIXELS}/(iw*ih))/2)*2"
    scale = f"scale='if(gt(iw*ih,{MAX_IMAGE_PIXELS}),{fit},iw)':-2"
    times = frame_times(duration)
    missing = 0
    for index, at in enumerate(times):
        frame = out_dir / f"{path.name}-frame-{index + 1:02d}.jpg"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{at:.2f}", "-i", str(path)]
            + ["-frames:v", "1", "-vf", scale, "-q:v", "3", "-y", str(frame)],
            capture_output=True,
            timeout=COMMAND_TIMEOUT,
        )
        if frame.is_file():
            blocks.append({"type": "text", "text": f"Frame at {at:.1f} s:"})
            blocks.append({"type": "image", "path": frame.name})
        else:
            missing += 1
    if missing:
        blocks.append(note(f"{missing} of {len(times)} frames could not be extracted."))
    return blocks


def parse_file(path: Path, out_dir: Path, work_dir: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return [text_block(path.read_text(errors="replace"))]
    if suffix == ".pdf":
        return parse_pdf(path, out_dir)
    if suffix == ".pptx":
        return parse_pptx(path, out_dir, work_dir)
    if suffix == ".xlsx":
        return parse_xlsx(path, out_dir, work_dir)
    if suffix in {".mp4", ".mov", ".webm", ".mkv"}:
        return parse_video(path, out_dir)
    return [note(f"Files of type {suffix or 'without an extension'} cannot be shown to the judge.")]


def main(submission_dir: str, output_dir: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[dict]] = {}
    for path in sorted(Path(submission_dir).iterdir()):
        if not path.is_file():
            continue
        with tempfile.TemporaryDirectory() as work:
            try:
                manifest[path.name] = parse_file(path, out_dir, Path(work))
            except subprocess.TimeoutExpired:
                # A stuck tool is the grader's failure, not the submission's.
                raise
            except Exception as e:
                failure = f"The file could not be parsed: {type(e).__name__}: {e}"
                manifest[path.name] = [note(failure)]
    (out_dir / "manifest.json").write_text(json.dumps({"files": manifest}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
