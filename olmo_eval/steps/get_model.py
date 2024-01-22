import logging
import os
from typing import Optional, Union

from tango import step

logger = logging.getLogger(__name__)


@step("get-model-path", cacheable=True, version="006")
def get_model_path(
    model_path: Union[str, os.PathLike],
    revision: Optional[str] = None,
) -> Union[str, os.PathLike]:
    """
    This is a temporary step that downloads the olmo checkpoint from the remote location
    and makes it compatible with HF model loading. In the future, when we have HF-compatible
    checkpoints from the get-go, this step will not be required.
    """
    assert "olmo" in str(model_path)
    try:
        model_dir = os.environ["GLOBAL_MODEL_DIR"]
    except KeyError:
        raise KeyError(
            "Please set `GLOBAL_MODEL_DIR` to some location locally accessible to your experiment run"
            ", like /net/nfs.cirrascale"
        )

    checkpoint_dir = str(model_path)
    if revision:
        checkpoint_dir += "/" + revision

    try:
        from hf_olmo.convert_olmo_to_hf import (
            download_remote_checkpoint_and_convert_to_hf,
        )

        local_model_path = download_remote_checkpoint_and_convert_to_hf(
            checkpoint_dir=checkpoint_dir, local_dir=model_dir
        )
    except ImportError:
        raise ImportError("Package `hf_olmo` cannot be found on the PYTHONPATH.")
        model_name = os.path.basename(checkpoint_dir)
        local_model_path = os.path.join(model_dir, model_name)

    return local_model_path
