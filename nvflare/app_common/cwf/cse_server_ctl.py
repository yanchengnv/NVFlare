# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Task
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.app_common.app_constant import AppConstants, ModelName
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.cwf.common import Constant, ModelType
from nvflare.app_common.cwf.server_ctl import ServerSideController
from nvflare.app_common.utils.val_result_manager import EvalResultManager


class CrossSiteEvalServerController(ServerSideController):
    def __init__(
        self,
        start_task_name=Constant.TASK_NAME_START,
        start_task_timeout=5,
        configure_task_name=Constant.TASK_NAME_CONFIGURE,
        configure_task_timeout=2,
        eval_task_name=Constant.TASK_NAME_EVAL,
        eval_task_timeout=30,
        task_check_period: float = 0.5,
        job_status_check_interval: float = 2.0,
        participating_clients=None,
        starting_client: str = None,
        evaluators=None,
        evaluatees=None,
        max_status_report_interval: float = 3600.0,
        eval_result_dir=AppConstants.CROSS_VAL_DIR,
        eval_local=True,
        eval_global=True,
    ):
        super().__init__(
            num_rounds=1,
            start_task_name=start_task_name,
            start_task_timeout=start_task_timeout,
            configure_task_name=configure_task_name,
            configure_task_timeout=configure_task_timeout,
            task_check_period=task_check_period,
            job_status_check_interval=job_status_check_interval,
            participating_clients=participating_clients,
            starting_client=starting_client,
            max_status_report_interval=max_status_report_interval,
        )

        self.eval_task_name = eval_task_name
        self.eval_task_timeout = eval_task_timeout
        self.eval_local = eval_local
        self.eval_global = eval_global
        self.evaluators = evaluators
        self.evaluatees = evaluatees

        if not self.eval_global and not self.eval_local:
            raise ValueError(f"nothing to eval")

        self.eval_result_dir = eval_result_dir
        self.global_names = []
        self.eval_manager = None
        self.current_round = 0

    def start_controller(self, fl_ctx: FLContext):
        super().start_controller(fl_ctx)

        if not self.evaluators:
            # every participating client is a train client
            self.evaluators = self.participating_clients
        else:
            for c in self.evaluators:
                if c not in self.participating_clients:
                    raise RuntimeError(f"Config Error: evaluator client {c} is not in participating_clients")

        if self.evaluatees == "none":
            # no evaluatees - this is the case that every client only evaluates global models
            if not self.eval_global:
                raise RuntimeError("Config Error: no evaluatees defined and no global model to evaluate!")
            self.evaluatees = []
        elif not self.evaluatees:
            # every participating client is an evaluatee
            self.evaluatees = self.participating_clients
        else:
            if not isinstance(self.evaluatees, list):
                raise RuntimeError("Config Error: evaluatees must be a list of client names")

            for c in self.evaluatees:
                if c not in self.participating_clients:
                    raise RuntimeError(f"Config Error: evaluatee client {c} is not in participating_clients")

        workspace: Workspace = self._engine.get_workspace()
        run_dir = workspace.get_run_dir(fl_ctx.get_job_id())
        cross_val_path = os.path.join(run_dir, self.eval_result_dir)
        cross_val_results_dir = os.path.join(cross_val_path, AppConstants.CROSS_VAL_RESULTS_DIR_NAME)
        self.eval_manager = EvalResultManager(cross_val_results_dir)

    def prepare_config(self):
        return {
            Constant.EVAL_LOCAL: self.eval_local,
            Constant.EVAL_GLOBAL: self.eval_global,
            Constant.EVALUATORS: self.evaluators,
            Constant.EVALUATEES: self.evaluatees,
        }

    def client_started(self, client_task: ClientTask, fl_ctx: FLContext):
        client = client_task.client
        result = client_task.result
        if self.eval_global:
            # get global model names from result
            global_names = result.get(Constant.GLOBAL_NAMES)
            if not global_names:
                self.log_error(fl_ctx, f"client {client.name} has no global models")
                return False
            self.global_names = global_names
            self.log_info(fl_ctx, f"got global model names from {client.name}: {global_names}")
        return True

    def _ask_to_evaluate(
        self, current_round: int, model_name: str, model_type: str, model_owner: str, fl_ctx: FLContext
    ):
        self.log_info(
            fl_ctx, f"Sending {model_name} model to all participating clients for validation round {current_round}"
        )

        # Create validation task and broadcast to all participating clients.
        task_data = Shareable()
        task_data[AppConstants.CURRENT_ROUND] = current_round
        task_data[Constant.MODEL_OWNER] = model_owner  # client that holds the model
        task_data[Constant.MODEL_NAME] = model_name
        task_data[Constant.MODEL_TYPE] = model_type

        task = Task(
            name=self.eval_task_name,
            data=task_data,
            result_received_cb=self._process_eval_result,
            timeout=self.eval_task_timeout,
        )

        self.broadcast(
            task=task,
            fl_ctx=fl_ctx,
            targets=self.participating_clients,
            min_responses=len(self.participating_clients),
            wait_time_after_min_received=0,
        )

    def sub_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        # ask everyone to evaluate global model
        for m in self.global_names:
            self._ask_to_evaluate(
                current_round=self.current_round,
                model_name=m,
                model_type=ModelType.GLOBAL,
                model_owner=self.starting_client,
                fl_ctx=fl_ctx,
            )
            self.current_round += 1

        # ask everyone to eval everyone else's local model
        if self.eval_local:
            for c in self.participating_clients:
                self._ask_to_evaluate(
                    current_round=self.current_round,
                    model_name=ModelName.BEST_MODEL,
                    model_type=ModelType.LOCAL,
                    model_owner=c,
                    fl_ctx=fl_ctx,
                )
                self.current_round += 1

    def is_sub_flow_done(self, fl_ctx: FLContext) -> bool:
        return self.get_num_standing_tasks() == 0

    def _process_eval_result(self, client_task: ClientTask, fl_ctx: FLContext):
        # Find name of the client sending this
        result = client_task.result
        client_name = client_task.client.name
        self._accept_eval_result(client_name=client_name, result=result, fl_ctx=fl_ctx)

    def _accept_eval_result(self, client_name: str, result: Shareable, fl_ctx: FLContext):
        model_owner = result.get_header(Constant.MODEL_OWNER, "")
        model_type = result.get_header(Constant.MODEL_TYPE)
        model_name = result.get_header(Constant.MODEL_NAME)

        if model_type == ModelType.GLOBAL:
            # global model
            model_owner = "GLOBAL_" + model_name
            model_info = model_owner
        else:
            model_info = f"{model_name} of {model_owner}"

        # Fire event. This needs to be a new local context per each client
        fl_ctx.set_prop(AppConstants.MODEL_OWNER, model_owner, private=True, sticky=False)
        fl_ctx.set_prop(AppConstants.DATA_CLIENT, client_name, private=True, sticky=False)
        fl_ctx.set_prop(AppConstants.VALIDATION_RESULT, result, private=True, sticky=False)
        self.fire_event(AppEventType.VALIDATION_RESULT_RECEIVED, fl_ctx)

        rc = result.get_return_code(ReturnCode.OK)
        if rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"bad evaluation result from client {client_name} on model {model_info}")
        else:
            dxo = from_shareable(result)
            location = self.eval_manager.add_result(evaluatee=model_owner, evaluator=client_name, result=dxo)
            self.log_info(fl_ctx, f"saved evaluation result from {client_name} on model {model_info} in {location}")
