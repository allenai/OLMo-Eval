import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tango.common.aliases import PathOrStr
from tango.common.logging import initialize_logging, teardown_logging
from tango.common.params import Params
from tango.common.testing import TangoTestCase
from tango.settings import TangoGlobalSettings


class LLMEvalTestCase(TangoTestCase):
    PROJECT_ROOT = (Path(__file__).parent / ".." / ".." / "..").resolve()
    """
    Root of the git repository.
    """

    # to run test suite with finished package, which does not contain
    # tests & fixtures, we must be able to look them up somewhere else
    PROJECT_ROOT_FALLBACK = (
        # users wanting to run test suite for installed package
        Path(os.environ["TANGO_SRC_DIR"])
        if "TANGO_SRC_DIR" in os.environ
        else (
            # fallback for conda packaging
            Path(os.environ["SRC_DIR"])
            if "CONDA_BUILD" in os.environ
            # stay in-tree
            else PROJECT_ROOT
        )
    )

    MODULE_ROOT = PROJECT_ROOT_FALLBACK / "tango"
    """
    Root of the tango module.
    """

    TESTS_ROOT = PROJECT_ROOT_FALLBACK / "tests"
    """
    Root of the tests directory.
    """

    FIXTURES_ROOT = PROJECT_ROOT_FALLBACK / "test_fixtures"
    """
    Root of the test fixtures directory.
    """


@contextmanager
def run_experiment(
    config: Union[PathOrStr, Dict[str, Any], Params],
    overrides: Optional[Union[Dict[str, Any], str]] = None,
    file_friendly_logging: bool = True,
    include_package: Optional[List[str]] = None,
    workspace_url: Optional[str] = None,
    parallelism: Optional[int] = 1,
    multicore: Optional[bool] = False,
    name: Optional[str] = None,
    settings: Optional[TangoGlobalSettings] = None,
):
    """
    A context manager to make testing experiments easier. On ``__enter__`` it runs
    the experiment and returns the path to the run directory, a temporary directory that will be
    cleaned up on ``__exit__``.
    """
    initialize_logging(enable_cli_logs=True, file_friendly_logging=file_friendly_logging)
    test_case = LLMEvalTestCase()
    try:
        test_case.setup_method()
        yield test_case.run(
            config,
            overrides=overrides,
            include_package=include_package,
            workspace_url=workspace_url,
            parallelism=parallelism,
            multicore=multicore,
            name=name,
            settings=settings,
        )
    finally:
        test_case.teardown_method()
        teardown_logging()
