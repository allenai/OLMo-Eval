## Description

<!-- What changed and why. For a new or ported eval, name the reference (oe-eval config,
paper, or repository revision) and any intentional deviations. -->

## Affected results

<!-- Which existing task specs or suites measure something different after this change,
if any. Changes to existing results need an entry in docs/result-compatibility.md. -->

## Type of Change

<!-- Put an `x` in all the boxes that apply -->

- [ ] Bug fix
- [ ] New task, variant, or suite
- [ ] New feature
- [ ] Breaking change or change to existing results
- [ ] Dependency or image update
- [ ] Documentation update
- [ ] Refactoring (no functional changes)

## Validation

<!-- Exact commands run, Beaker experiment IDs, and reference comparisons as applicable.
List checks that were not run under remaining work rather than here. -->

- [ ] `uv run ruff check src/ tests/` and `uv run ruff format --check src/ tests/`
- [ ] `uv run ty check src/`
- [ ] `uv run pytest tests/ --ignore=tests/integration -v`
- [ ] Integration tests, if storage, providers, or sandboxes changed
- [ ] New tests added for new behavior

## Remaining work

<!-- Known gaps, deferred work, and links to follow-up issues. -->

See [CONTRIBUTING.md](https://github.com/allenai/olmo-eval/blob/main/CONTRIBUTING.md#pull-requests) for PR expectations.
