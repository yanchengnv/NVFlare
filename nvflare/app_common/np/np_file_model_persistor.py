# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.security.logging import secure_format_exception

from .constants import NPConstants


def _get_run_dir(fl_ctx: FLContext):
    job_id = fl_ctx.get_job_id()
    workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
    run_dir = workspace.get_run_dir(job_id)
    return run_dir


class NPFileModelPersistor(ModelPersistor):
    def __init__(
        self,
        last_global_model_file_name="last_global_model.npy",
        best_global_model_file_name="best_global_model.npy",
        model_dir="models",
        initial_model_file_name="initial_model.npy",
    ):
        super().__init__()

        self.model_dir = model_dir
        self.last_global_model_file_name = last_global_model_file_name
        self.best_global_model_file_name = best_global_model_file_name
        self.initial_model_file_name = initial_model_file_name

        # This is default model that will be used if not local model is provided.
        self.default_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        run_dir = _get_run_dir(fl_ctx)
        model_path = os.path.join(run_dir, self.model_dir, self.initial_model_file_name)
        try:
            # try loading previous model
            data = np.load(model_path)
        except Exception as e:
            self.log_info(
                fl_ctx,
                f"Unable to load model from {model_path}: {secure_format_exception(e)}. Using default data instead.",
                fire_event=False,
            )
            data = self.default_data.copy()

        model_learnable = make_model_learnable(weights={NPConstants.NUMPY_KEY: data}, meta_props={})
        self.log_info(fl_ctx, f"Loaded initial model: {model_learnable}")
        return model_learnable

    def save_model(self, model_learnable: ModelLearnable, fl_ctx: FLContext):
        self._save(fl_ctx, model_learnable, self.last_global_model_file_name)

    def _save(self, fl_ctx: FLContext, model_learnable: ModelLearnable, file_name: str):
        run_dir = _get_run_dir(fl_ctx)
        model_root_dir = os.path.join(run_dir, self.model_dir)
        if not os.path.exists(model_root_dir):
            os.makedirs(model_root_dir)

        model_path = os.path.join(model_root_dir, file_name)
        np.save(model_path, model_learnable[ModelLearnableKey.WEIGHTS][NPConstants.NUMPY_KEY])
        self.log_info(fl_ctx, f"Saved numpy model to: {model_path}")
        self.log_info(fl_ctx, f"Model: {model_learnable}")

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE:
            # save the current model as the best model!
            model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
            self._save(fl_ctx, model, self.best_global_model_file_name)
