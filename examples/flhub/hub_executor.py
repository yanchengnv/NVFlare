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

from nvflare.apis.dxo import from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.fuel.utils.pipe.pipe import Pipe
from .defs import Topic, send_to_pipe, receive_from_pipe

import time


class HubExecutor(Executor):

    def __init__(self,
                 pipe_id: str,
                 task_wait_time: float,
                 result_poll_interval: float=0.5):
        Executor.__init__(self)
        self.pipe_id = pipe_id
        self.task_wait_time = task_wait_time
        self.result_poll_interval = result_poll_interval
        self.pipe = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        if event_type == EventType.START_RUN:
            job_id = fl_ctx.get_job_id()
            self.pipe = engine.get_component(self.pipe_id)
            if not isinstance(self.pipe, Pipe):
                raise TypeError(f"pipe must be Pipe type. Got: {type(self.pipe)}")
            self.pipe.open(name=job_id, me='x')
        elif event_type == EventType.ABORT_TASK:
            send_to_pipe(self.pipe, topic=Topic.ABORT_TASK, data="")
        elif event_type == EventType.END_RUN:
            send_to_pipe(self.pipe, topic=Topic.END_RUN, data="")

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"Client trainer got task: {task_name}")

        # send the task to T2
        self.log_info(fl_ctx, f"sent task data to T2 for task {task_name}")
        send_to_pipe(self.pipe, topic=task_name, data=shareable)

        # wait for result from T2
        start = time.time()
        while True:
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            topic, data = receive_from_pipe(self.pipe)
            if not topic:
                if self.task_wait_time and time.time() - start > self.task_wait_time:
                    # timed out
                    self.log_error(fl_ctx,
                                   f"task '{task_name}' timeout after {self.task_wait_time} secs")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            elif topic == Topic.END_RUN:
                self.log_error(fl_ctx,
                               f"received {topic} from T2 while waiting for task {task_name}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            elif topic != task_name:
                # ignore wrong task name
                self.log_error(fl_ctx, f"ignored '{topic}' when waiting for '{task_name}'")
            else:
                self.log_info(fl_ctx, f"got result for task '{topic}' from T2")
                if not isinstance(data, Shareable):
                    self.log_error(fl_ctx,
                                   f"bad result data from T2 - must be Shareable butt got {type(data)}")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
                dxo = from_shareable(data)
                return dxo.to_shareable()
            time.sleep(self.result_poll_interval)
