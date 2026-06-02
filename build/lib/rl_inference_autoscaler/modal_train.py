"""Backward-compatible entrypoint; implementation lives in ``cloud.modal_app``."""

from rl_inference_autoscaler.cloud.modal_app import *  # noqa: F403
from rl_inference_autoscaler.cloud.modal_app import (  # noqa: F401
    APP_NAME,
    app,
    get_app,
    main,
    train_on_modal,
)
