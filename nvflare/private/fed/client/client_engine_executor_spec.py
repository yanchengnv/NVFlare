# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import time
from abc import ABC, abstractmethod

from nvflare.apis.client_engine_spec import ClientEngineSpec
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.workspace import Workspace
from nvflare.widgets.widget import Widget


class TaskAssignment(object):
    def __init__(self, name: str, task_id: str, data: Shareable):
        """Init TaskAssignment.

        Keeps track of information about the assignment of a task, including the time
        that it was created after being fetched by the Client Run Manager.

        Args:
            name: task name
            task_id: task id
            data: the Shareable data for the task assignment
        """
        self.name = name
        self.task_id = task_id
        self.data = data
        self.receive_time = time.time()


class ClientEngineExecutorSpec(ClientEngineSpec, ABC):
    """The ClientEngineExecutorSpec defines the ClientEngine APIs running in the child process."""

    @abstractmethod
    def get_task_assignment(self, fl_ctx: FLContext) -> TaskAssignment:
        pass

    @abstractmethod
    def send_task_result(self, result: Shareable, fl_ctx: FLContext) -> bool:
        pass

    @abstractmethod
    def get_all_components(self) -> dict:
        pass

    @abstractmethod
    def build_component(self, config_dict):
        """Build a component from the config_dict.

        Args:
            config_dict: config dict

        """

    @abstractmethod
    def abort_app(self, job_id: str, fl_ctx: FLContext):
        """Abort the running FL App on the client.

        Args:
            job_id: current_job_id
            fl_ctx: FLContext

        """
        pass
