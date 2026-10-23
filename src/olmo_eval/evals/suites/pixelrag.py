"""PixelRAG reader suite: Wikipedia QA answered from retrieved screenshot tiles."""

from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

PIXELRAG_TASKS = (
    "pixelrag_simpleqa",
    "pixelrag_nq",
    "pixelrag_nq_tables",
    "pixelrag_mmsearch",
    "pixelrag_evqa",
)

make_suite(
    "pixelrag",
    PIXELRAG_TASKS,
    # The paper reports each benchmark on its own; no cross-task average.
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description=(
        "PixelRAG reader benchmark: SimpleQA, NQ, NQ-Tables, MMSearch and Encyclopedic-VQA "
        "landmarks answered from the top 3 retrieved Wikipedia screenshots, GPT-4.1 judged."
    ),
)
