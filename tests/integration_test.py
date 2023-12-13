import json
import os

import petname
from tango.common.params import Params

from olmo_eval.common.testing import LLMEvalTestCase, run_experiment


def test_jsonnet_config():
    run_name = petname.generate()
    config_path = LLMEvalTestCase.FIXTURES_ROOT / "test_config.jsonnet"
    config = Params.from_file(config_path)
    os.environ["EVAL_DATA_PATH"] = str(LLMEvalTestCase.FIXTURES_ROOT)
    with run_experiment(
        config,
        name=run_name,
        multicore=None,
    ) as run_path:
        step_info_path = run_path / "stepinfo.json"
        with open(step_info_path) as f:
            infos = json.load(f)
        num_executed = len(infos)
        assert num_executed == 10
