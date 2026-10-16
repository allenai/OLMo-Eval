"""The vision benchmarks; importing this package registers them.

One module per benchmark. Imports are explicit (unlike ``evals/tasks``\'
pkgutil scan) so a missing module is an ImportError here rather than a silently
absent task.
"""

from olmo_eval.evals.vision.benchmarks import ai2d as _ai2d  # noqa: F401
from olmo_eval.evals.vision.benchmarks import blink as _blink  # noqa: F401
from olmo_eval.evals.vision.benchmarks import cc_ocr as _cc_ocr  # noqa: F401
from olmo_eval.evals.vision.benchmarks import chart_qa as _chart_qa  # noqa: F401
from olmo_eval.evals.vision.benchmarks import charxiv as _charxiv  # noqa: F401
from olmo_eval.evals.vision.benchmarks import countbench_qa as _countbench_qa  # noqa: F401
from olmo_eval.evals.vision.benchmarks import dense_caption as _dense_caption  # noqa: F401
from olmo_eval.evals.vision.benchmarks import doc_qa as _doc_qa  # noqa: F401
from olmo_eval.evals.vision.benchmarks import info_qa as _info_qa  # noqa: F401
from olmo_eval.evals.vision.benchmarks import math_vista as _math_vista  # noqa: F401
from olmo_eval.evals.vision.benchmarks import mmiu as _mmiu  # noqa: F401
from olmo_eval.evals.vision.benchmarks import mmmu as _mmmu  # noqa: F401
from olmo_eval.evals.vision.benchmarks import mmmu_pro as _mmmu_pro  # noqa: F401
from olmo_eval.evals.vision.benchmarks import muir_bench as _muir_bench  # noqa: F401
from olmo_eval.evals.vision.benchmarks import olmocr_bench as _olmocr_bench  # noqa: F401
from olmo_eval.evals.vision.benchmarks import omnidocbench as _omnidocbench  # noqa: F401
from olmo_eval.evals.vision.benchmarks import pixmo_count as _pixmo_count  # noqa: F401
from olmo_eval.evals.vision.benchmarks import pixmo_points_eval as _pixmo_points_eval  # noqa: F401
from olmo_eval.evals.vision.benchmarks import real_world_qa as _real_world_qa  # noqa: F401
from olmo_eval.evals.vision.benchmarks import sa_co_gold as _sa_co_gold  # noqa: F401
from olmo_eval.evals.vision.benchmarks import text_vqa as _text_vqa  # noqa: F401
from olmo_eval.evals.vision.benchmarks import vqa2 as _vqa2  # noqa: F401

__all__: list[str] = []
