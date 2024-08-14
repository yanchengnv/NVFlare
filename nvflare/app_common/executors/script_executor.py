# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional

from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.in_process_client_api_executor import InProcessClientAPIExecutor
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.import_utils import optional_import

torch, torch_ok = optional_import(module="torch")
if torch_ok:
    from nvflare.app_opt.pt.decomposers import TensorDecomposer
    from nvflare.app_opt.pt.params_converter import NumpyToPTParamsConverter, PTToNumpyParamsConverter

    DEFAULT_PARAMS_EXCHANGE_FORMAT = ExchangeFormat.PYTORCH
else:
    DEFAULT_PARAMS_EXCHANGE_FORMAT = ExchangeFormat.NUMPY

tensorflow, tf_ok = optional_import(module="tensorflow")
if tf_ok:
    from nvflare.app_opt.tf.params_converter import KerasModelToNumpyParamsConverter, NumpyToKerasModelParamsConverter


class ScriptExecutor(InProcessClientAPIExecutor):
    def __init__(
        self,
        task_script_path: str,
        task_script_args: str = "",
        task_wait_time: Optional[float] = None,
        result_pull_interval: float = 0.5,
        log_pull_interval: Optional[float] = None,
        params_transfer_type: TransferType = TransferType.FULL,
        from_nvflare_converter_id: Optional[str] = None,
        to_nvflare_converter_id: Optional[str] = None,
        train_with_evaluation: bool = True,
        train_task_name: str = "train",
        evaluate_task_name: str = "evaluate",
        submit_model_task_name: str = "submit_model",
        params_exchange_format=DEFAULT_PARAMS_EXCHANGE_FORMAT,
    ):
        """Wrapper around InProcessClientAPIExecutor for different params_exchange_format. Currently defaulting to `params_exchange_format=ExchangeFormat.PYTORCH`.

        Args:
        """
        super(ScriptExecutor, self).__init__(
            task_script_path=task_script_path,
            task_script_args=task_script_args,
            task_wait_time=task_wait_time,
            result_pull_interval=result_pull_interval,
            train_with_evaluation=train_with_evaluation,
            train_task_name=train_task_name,
            evaluate_task_name=evaluate_task_name,
            submit_model_task_name=submit_model_task_name,
            from_nvflare_converter_id=from_nvflare_converter_id,
            to_nvflare_converter_id=to_nvflare_converter_id,
            params_exchange_format=params_exchange_format,
            params_transfer_type=params_transfer_type,
            log_pull_interval=log_pull_interval,
        )
        self.task_script_path = task_script_path
        if torch_ok:
            if params_exchange_format == ExchangeFormat.PYTORCH:
                fobs.register(TensorDecomposer)

                if self._from_nvflare_converter is None:
                    self._from_nvflare_converter = NumpyToPTParamsConverter(
                        [AppConstants.TASK_TRAIN, AppConstants.TASK_VALIDATION]
                    )
                if self._to_nvflare_converter is None:
                    self._to_nvflare_converter = PTToNumpyParamsConverter(
                        [AppConstants.TASK_TRAIN, AppConstants.TASK_SUBMIT_MODEL]
                    )
        if tf_ok:
            if params_exchange_format == ExchangeFormat.NUMPY:
                if self._from_nvflare_converter is None:
                    self._from_nvflare_converter = NumpyToKerasModelParamsConverter(
                        [AppConstants.TASK_TRAIN, AppConstants.TASK_VALIDATION]
                    )
                if self._to_nvflare_converter is None:
                    self._to_nvflare_converter = KerasModelToNumpyParamsConverter(
                        [AppConstants.TASK_TRAIN, AppConstants.TASK_SUBMIT_MODEL]
                    )
        # TODO: support other params_exchange_format

    def add_to_fed_job(self, job, ctx, **kwargs):
        """This method is required by Job API.

        Args:
            job: the Job object to add to
            ctx: Job Context

        Returns:

        """
        super().add_to_fed_job(job, ctx, **kwargs)
        job.add_resources(resources=[self.task_script_path], ctx=ctx)
