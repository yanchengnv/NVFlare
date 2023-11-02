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

import logging
import threading
import time
import traceback

from nvflare.apis.dxo import DXO
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.app_common.abstract.exchange_task import ExchangeTask
from nvflare.app_common.decomposers import common_decomposers
from nvflare.fuel.utils.constants import PipeChannelName
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.fuel.utils.pipe.pipe import Message, Mode, Pipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler


class RC:
    OK = "OK"
    BAD_TASK_DATA = "BAD_TASK_DATA"
    EXECUTION_EXCEPTION = "EXECUTION_EXCEPTION"


class AgentClosed(Exception):
    pass


class CallStateError(Exception):
    pass


class Task:
    def __init__(self, task_name: str, task_id: str, dxo: DXO):
        self.task_name = task_name
        self.task_id = task_id
        self.dxo = dxo

    def __str__(self):
        return f"'{self.task_name} {self.task_id}'"


class _TaskContext:
    def __init__(self, task_id, task_name: str, msg_id):
        self.task_id = task_id
        self.task_name = task_name
        self.msg_id = msg_id


class FlareAgent:
    def __init__(
        self,
        pipe: Pipe,
        read_interval=0.1,
        heartbeat_interval=5.0,
        heartbeat_timeout=30.0,
        resend_interval=2.0,
        max_resends=None,
        submit_result_timeout=30.0,
        metric_pipe=None,
        task_channel_name=PipeChannelName.TASK,
        metric_channel_name=PipeChannelName.METRIC,
    ):
        """Constructor of Flare Agent. The agent is responsible for communicating with the Flare Client Job cell (CJ)
        to get task and to submit task result.

        Args:
            pipe: pipe for communication
            submit_result_timeout: when submitting task result, how long to wait for response from the CJ
        """
        flare_decomposers.register()
        common_decomposers.register()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.pipe = pipe
        self.pipe_handler = PipeHandler(
            pipe=self.pipe,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            resend_interval=resend_interval,
            max_resends=max_resends,
        )
        self.submit_result_timeout = submit_result_timeout
        self.task_channel_name = task_channel_name
        self.metric_channel_name = metric_channel_name

        self.metric_pipe = metric_pipe
        self.metric_pipe_handler = None
        if self.metric_pipe:
            self.metric_pipe_handler = PipeHandler(
                pipe=self.metric_pipe,
                read_interval=read_interval,
                heartbeat_interval=heartbeat_interval,
                heartbeat_timeout=heartbeat_timeout,
                resend_interval=resend_interval,
                max_resends=max_resends,
            )

        self.current_task = None
        self.task_lock = threading.Lock()
        self.asked_to_stop = False

    def start(self):
        """Start the agent. This method must be called to enable CJ/Agent communication.

        Returns: None

        """
        self.pipe.open(self.task_channel_name)
        self.pipe_handler.set_status_cb(self._status_cb, pipe_handler=self.pipe_handler, channel=self.task_channel_name)
        self.pipe_handler.start()

        if self.metric_pipe:
            self.metric_pipe.open(self.metric_channel_name)
            self.metric_pipe_handler.set_status_cb(
                self._status_cb, pipe_handler=self.metric_pipe_handler, channel=self.metric_channel_name
            )
            self.metric_pipe_handler.start()

    def _status_cb(self, msg: Message, pipe_handler: PipeHandler, channel):
        self.logger.info(f"{channel} pipe status changed to {msg.topic}: {msg.data}")
        self.asked_to_stop = True
        pipe_handler.stop()

    def stop(self):
        """Stop the agent. After this is called, there will be no more communications between CJ and agent.

        Returns: None

        """
        self.asked_to_stop = True
        self.pipe_handler.stop()

    def get_task(self):
        """Get a task from FLARE. This is a blocking call.

        If timeout is specified, this call is blocked only for the specified amount of time.
        If timeout is not specified, this call is blocked forever until a task is received or agent is closed.

        Returns: None if no task is available during before timeout; or a Task object if task is available.
        Raises:
            AgentClosed exception if the agent is closed before timeout.
            CallStateError exception if the call is not made properly.

        Note: the application must make the call only when it is just started or after a previous task's result
        has been submitted.

        """
        while True:
            if self.asked_to_stop:
                raise AgentClosed("agent closed")

            if self.current_task:
                raise CallStateError(f"application called get_task while the current task is not processed")

            req = self.pipe_handler.get_next()
            if req:
                if not isinstance(req.data, ExchangeTask):
                    self.logger.error(f"bad task: expect request data to be ExchangeTask but got {type(req.data)}")
                    raise RuntimeError("bad request data")

                ex_task = req.data
                if not isinstance(ex_task.data, DXO):
                    self.logger.error(f"bad task: expect task data to be DXO but got {type(ex_task.data)}")
                    raise RuntimeError("bad task data")

                tc = _TaskContext(
                    task_id=ex_task.task_id,
                    task_name=ex_task.task_name,
                    msg_id=req.msg_id,
                )
                self.current_task = tc
                return Task(task_name=tc.task_name, task_id=tc.task_id, dxo=ex_task.data)
            time.sleep(0.5)

    def submit_result(self, result: DXO, rc=RC.OK) -> bool:
        """Submit the result of the current task.
        This is a blocking call. The agent will try to send the result to flare site until it is successfully sent or
        the task is aborted or the agent is closed.

        Args:
            result: result to be submitted
            rc: return code

        Returns: whether the result is submitted successfully
        Raises: the CallStateError exception if the submit_result call is not made properly.

        Notes: the application must only make this call after the received task is processed. The call can only be
        made a single time regardless whether the submission is successful.

        """
        if result and not isinstance(result, DXO):
            raise TypeError(f"result must be DXO but got {type(result)}")

        with self.task_lock:
            current_task = self.current_task
            if not current_task:
                self.logger.error("submit_result is called but there is no current task!")
                return False

        try:
            result = self._do_submit_result(current_task, result, rc)
        except:
            self.logger.error(f"exception submitting result to {current_task.sender}")
            traceback.print_exc()
            result = False

        with self.task_lock:
            self.current_task = None

        return result

    def _do_submit_result(self, current_task: _TaskContext, result: DXO, rc):
        ex_task = ExchangeTask(
            task_name=current_task.task_name, task_id=current_task.task_id, data=result, return_code=rc
        )
        reply = Message.new_reply(topic=current_task.task_name, req_msg_id=current_task.msg_id, data=ex_task)
        return self.pipe_handler.send_to_peer(reply, self.submit_result_timeout)

    def log_metric(self, record: DXO):
        if not self.metric_pipe_handler:
            raise RuntimeError("metric pipe is not available")

        msg = Message.new_request(topic="metric", data=record)
        return self.metric_pipe_handler.send_to_peer(msg, self.submit_result_timeout)


class FlareAgentWithCellPipe(FlareAgent):
    def __init__(
        self,
        agent_id: str,
        site_name: str,
        root_url: str,
        secure_mode: bool,
        workspace_dir: str,
        read_interval=0.1,
        heartbeat_interval=5.0,
        heartbeat_timeout=30.0,
        resend_interval=2.0,
        max_resends=None,
        submit_result_timeout=30.0,
        has_metrics=False,
    ):
        pipe = CellPipe(
            mode=Mode.ACTIVE,
            token=agent_id,
            site_name=site_name,
            root_url=root_url,
            secure_mode=secure_mode,
            workspace_dir=workspace_dir,
        )

        metric_pipe = None
        if has_metrics:
            metric_pipe = CellPipe(
                mode=Mode.ACTIVE,
                token=agent_id,
                site_name=site_name,
                root_url=root_url,
                secure_mode=secure_mode,
                workspace_dir=workspace_dir,
            )

        super().__init__(
            pipe,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            resend_interval=resend_interval,
            max_resends=max_resends,
            submit_result_timeout=submit_result_timeout,
            metric_pipe=metric_pipe,
        )
