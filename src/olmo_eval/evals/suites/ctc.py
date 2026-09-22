"""Suites for the CTC long-context family: one flag instead of 22 (or 177).

Two independent axes cut the suite, and the names keep them separate:

*Context length* -- how long the prompt is:

* ``ctc:figure``   -- every task at its 2k-32k ladder (the paper grid; 108 runs)
* ``ctc:xlong``    -- every task's rungs ABOVE 32k (64k-1M where built; 69 runs)
* ``ctc:r32k`` ... -- every task that has that rung, at that rung (one context-length column)

*Corpus-tracking demand* -- how much of the corpus must be held at once, the axis the suite is
named for (see :class:`olmo_eval.evals.tasks.ctc_suite.CTCClass`):

* ``ctc:low``      -- the 12 O(N) rows: an answer-bearing document exists and retrieval finds it
* ``ctc:high``     -- the 10 O(N^2)/O(NM)/O(N^3) rows: the answer is a relation over documents

The two axes compose, so the length-restricted forms are registered as well -- ``ctc:low:figure``,
``ctc:high:figure``, ``ctc:low:xlong``, ``ctc:high:xlong``. ``ctc:high:figure`` is the cheap
diagnostic worth reaching for first: it is where a long-context model that merely retrieves well
separates from one that actually tracks a corpus, at grid-sized cost.

And the rest:

* ``ctc:nq`` ...   -- one task's full ladder, every rung (one row)
* ``ctc``          -- everything (177 runs; know what you are asking for)

Aggregation is DISPLAY_ONLY throughout, including for ``ctc:low``/``ctc:high``: the tasks carry
heterogeneous metrics (f1, pair f1, kendall tau, ce_pos_recall, partial credit), and averaging
those into one number would be exactly the kind of quiet nonsense the per-task metric declarations
exist to prevent. A low-vs-high *comparison* is read per task, or as a gap on a shared metric --
never as one mean against another.
"""

from __future__ import annotations

from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, register
from olmo_eval.evals.tasks.ctc_suite import OOD_ROSTER, ROSTER, RUNG_TOKENS, CTCClass

_DISPLAY = AggregationStrategy.DISPLAY_ONLY

_BASE = tuple(r for r in RUNG_TOKENS if RUNG_TOKENS[r] <= 32768)
_XLONG = tuple(r for r in RUNG_TOKENS if RUNG_TOKENS[r] > 32768)

_FIGURE = register(
    Suite(
        name="ctc:figure",
        tasks=tuple(
            f"{name}:{rung}" for name, row in ROSTER.items() for rung in row.rungs if rung in _BASE
        ),
        aggregation=_DISPLAY,
        description="All 22 CTC tasks over the 2k-32k figure ladder.",
    )
)

_XLONG_SUITE = register(
    Suite(
        name="ctc:xlong",
        tasks=tuple(
            f"{name}:{rung}" for name, row in ROSTER.items() for rung in row.rungs if rung in _XLONG
        ),
        aggregation=_DISPLAY,
        description="Every task's rungs above 32k (64k-1M where the source corpus allows).",
    )
)

for _rung in RUNG_TOKENS:
    register(
        Suite(
            name=f"ctc:{_rung}",
            tasks=tuple(f"{name}:{_rung}" for name, row in ROSTER.items() if _rung in row.rungs),
            aggregation=_DISPLAY,
            description=f"Every CTC task that has a {_rung} rung, at {_rung}.",
        )
    )

for _name, _row in ROSTER.items():
    register(
        Suite(
            name=f"ctc:{_name.removeprefix('ctc_')}",
            tasks=tuple(f"{_name}:{rung}" for rung in _row.rungs),
            aggregation=_DISPLAY,
            description=f"The full {_name} ladder ({len(_row.rungs)} rungs).",
        )
    )

#: ``ctc_class`` -> (suite-name fragment, what the rows have in common, the classes they cover).
_CLASS_BLURB = {
    CTCClass.LOW: (
        "an answer-bearing document exists and the work is finding it",
        "O(N)",
    ),
    CTCClass.HIGH: (
        "the answer is a relation over documents, with no single span to retrieve",
        "O(N^2)/O(NM)/O(N^3)",
    ),
}

for _cls, (_blurb, _notation) in _CLASS_BLURB.items():
    _rows = {name: row for name, row in ROSTER.items() if row.ctc_class is _cls}
    register(
        Suite(
            name=f"ctc:{_cls.value}",
            tasks=tuple(f"{name}:{rung}" for name, row in _rows.items() for rung in row.rungs),
            aggregation=_DISPLAY,
            description=(f"The {len(_rows)} {_notation} CTC rows, every rung -- {_blurb}."),
        )
    )
    for _span, _rungs, _label in (
        ("figure", _BASE, "2k-32k figure ladder"),
        ("xlong", _XLONG, "rungs above 32k"),
    ):
        register(
            Suite(
                name=f"ctc:{_cls.value}:{_span}",
                tasks=tuple(
                    f"{name}:{rung}"
                    for name, row in _rows.items()
                    for rung in row.rungs
                    if rung in _rungs
                ),
                aggregation=_DISPLAY,
                description=f"The {len(_rows)} {_notation} CTC rows over the {_label}.",
            )
        )

#: The held-out rows, selectable on their own and by length. They are deliberately NOT folded into
#: ``ctc``/``ctc:figure``/``ctc:low``/``ctc:high``: those names select the frozen 22-row roster, and
#: quietly widening them would change what an already-published "figure ladder" number refers to.
#: Ask for OOD explicitly.
register(
    Suite(
        name="ctc:ood",
        tasks=tuple(f"{name}:{rung}" for name, row in OOD_ROSTER.items() for rung in row.rungs),
        aggregation=_DISPLAY,
        description=(
            f"The {len(OOD_ROSTER)} held-out CTC rows, every rung -- same graders as their "
            "in-distribution twins over corpora no SFT mix contains."
        ),
    )
)
for _span, _rungs, _label in (
    ("figure", _BASE, "2k-32k figure ladder"),
    ("xlong", _XLONG, "rungs above 32k"),
):
    register(
        Suite(
            name=f"ctc:ood:{_span}",
            tasks=tuple(
                f"{name}:{rung}"
                for name, row in OOD_ROSTER.items()
                for rung in row.rungs
                if rung in _rungs
            ),
            aggregation=_DISPLAY,
            description=f"The {len(OOD_ROSTER)} held-out CTC rows over the {_label}.",
        )
    )

for _name, _row in OOD_ROSTER.items():
    register(
        Suite(
            name=f"ctc:{_name.removeprefix('ctc_')}",
            tasks=tuple(f"{_name}:{rung}" for rung in _row.rungs),
            aggregation=_DISPLAY,
            description=f"The full {_name} ladder ({len(_row.rungs)} rungs).",
        )
    )

register(
    Suite(
        name="ctc",
        tasks=(_FIGURE, _XLONG_SUITE),  # Suite objects: string entries would be read as task names
        aggregation=_DISPLAY,
        description="The entire CTC suite: 22 tasks x every built rung.",
    )
)
