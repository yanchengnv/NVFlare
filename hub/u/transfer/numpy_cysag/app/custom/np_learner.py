import os

import numpy as np

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.np.constants import NPConstants


class NPLearner(Learner):
    def __init__(self):
        Learner.__init__(self)
        self._model_name = "best_numpy.npy"
        self._model_dir = "model"
        self._delta = 1

    def train(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # First we extract DXO from the shareable.
        try:
            incoming_dxo = from_shareable(data)
        except BaseException as e:
            self.system_panic(f"Unable to convert shareable to model definition. Exception {e}", fl_ctx)
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # Information about workflow is retrieved from the shareable header.
        current_round = data.get_header(AppConstants.CURRENT_ROUND, None)
        total_rounds = data.get_header(AppConstants.NUM_ROUNDS, None)

        # Ensure that data is of type weights. Extract model data.
        if incoming_dxo.data_kind != DataKind.WEIGHTS:
            self.system_panic("Model DXO should be of kind DataKind.WEIGHTS.", fl_ctx)
            return make_reply(ReturnCode.BAD_TASK_DATA)

        np_data = incoming_dxo.data

        # Display properties.
        self.log_info(fl_ctx, f"Incoming data kind: {incoming_dxo.data_kind}")
        self.log_info(fl_ctx, f"Model: \n{np_data}")
        self.log_info(fl_ctx, f"Current Round: {current_round}")
        self.log_info(fl_ctx, f"Total Rounds: {total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Doing some dummy training.
        if np_data:
            if NPConstants.NUMPY_KEY in np_data:
                np_data[NPConstants.NUMPY_KEY] += self._delta
            else:
                self.log_error(fl_ctx, "numpy_key not found in model.")
                return make_reply(ReturnCode.BAD_TASK_DATA)
        else:
            self.log_error(fl_ctx, "No model weights found in shareable.")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # We check abort_signal regularly to make sure
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Save local numpy model
        try:
            self._save_local_model(fl_ctx, np_data)
        except Exception as e:
            self.log_error(fl_ctx, f"Exception in saving local model: {e}.")

        self.log_info(
            fl_ctx,
            f"Model after training: {np_data}",
        )

        # Checking abort signal again.
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Prepare a DXO for our updated model. Create shareable and return
        outgoing_dxo = DXO(data_kind=incoming_dxo.data_kind, data=np_data, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1})
        return outgoing_dxo.to_shareable()

    def _save_local_model(self, fl_ctx: FLContext, model: dict):
        # Save local model
        engine = fl_ctx.get_engine()
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        run_dir = engine.get_workspace().get_run_dir(job_id)
        model_path = os.path.join(run_dir, self._model_dir)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_save_path = os.path.join(model_path, self._model_name)
        np.save(model_save_path, model[NPConstants.NUMPY_KEY])
        self.log_info(fl_ctx, f"Saved numpy model to: {model_save_path}")

    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
        # Retrieve the local model saved during training.
        np_data = None
        try:
            np_data = self._load_local_model(fl_ctx)
        except Exception as e:
            self.log_error(fl_ctx, f"Unable to load model: {e}")

        # Create DXO and shareable from model data.
        model_shareable = Shareable()
        if np_data:
            outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=np_data)
            model_shareable = outgoing_dxo.to_shareable()
        else:
            # Set return code.
            self.log_error(fl_ctx, "local model not found.")
            model_shareable.set_return_code(ReturnCode.EXECUTION_RESULT_ERROR)

        return model_shareable

    def _load_local_model(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        run_dir = engine.get_workspace().get_run_dir(job_id)
        model_path = os.path.join(run_dir, self._model_dir)

        model_load_path = os.path.join(model_path, self._model_name)
        try:
            np_data = np.load(model_load_path)
        except Exception as e:
            self.log_error(fl_ctx, f"Unable to load local model: {e}")
            return None

        model = ModelLearnable()
        model[NPConstants.NUMPY_KEY] = np_data

        return model
