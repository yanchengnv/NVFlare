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
import time
from abc import ABC, abstractmethod
from typing import Any

from nvflare.apis.dxo import DXO, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.exchange_task import ExchangeTask
from nvflare.app_common.app_constant import AppConstants
from nvflare.fuel.utils.constants import PipeChannelName
from nvflare.fuel.utils.pipe.pipe import Message, Pipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler
from nvflare.fuel.utils.validation_utils import check_non_negative_int, check_positive_number, check_str
from nvflare.security.logging import secure_format_exception


class TaskExchanger(Executor, ABC):
    def __init__(
        self,
        pipe_id: str,
        read_interval=0.1,
        heartbeat_interval=5.0,
        heartbeat_timeout=30.0,
        resend_interval=2.0,
        max_resends=None,
        peer_read_timeout=5.0,
        task_wait_time=None,
        result_poll_interval=0.5,
        pipe_channel_name=PipeChannelName.TASK,
    ):
        """Constructor of TaskExchanger

        Args:
            pipe_id: component id of pipe
            read_interval: how often to read from pipe
            heartbeat_interval: how often to send heartbeat to peer
            heartbeat_timeout: max amount of time to allow missing heartbeats before treating peer as dead
            resend_interval: how often to resend a message when failing to send
            max_resends: max number of resends. None means no limit
            peer_read_timeout: time to wait for peer to accept sent message
            task_wait_time: how long to wait for a task to complete. None means waiting forever
            result_poll_interval: how often to poll task result
            pipe_channel_name: the channel name for sending task requests
        """
        Executor.__init__(self)
        check_str("pipe_id", pipe_id)
        check_positive_number("read_interval", read_interval)
        check_positive_number("heartbeat_interval", heartbeat_interval)
        check_positive_number("heartbeat_timeout", heartbeat_timeout)
        check_positive_number("resend_interval", resend_interval)
        if max_resends is not None:
            check_non_negative_int("max_resends", max_resends)
        check_positive_number("peer_read_timeout", peer_read_timeout)
        if task_wait_time is not None:
            check_positive_number("task_wait_time", task_wait_time)
        check_positive_number("result_poll_interval", result_poll_interval)
        check_str("pipe_channel_name", pipe_channel_name)

        self.pipe_id = pipe_id
        self.read_interval = read_interval
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.resend_interval = resend_interval
        self.max_resends = max_resends
        self.peer_read_timeout = peer_read_timeout
        self.task_wait_time = task_wait_time
        self.result_poll_interval = result_poll_interval
        self.pipe_channel_name = pipe_channel_name
        self.pipe = None
        self.pipe_handler = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            self.pipe = engine.get_component(self.pipe_id)
            if not isinstance(self.pipe, Pipe):
                self.system_panic(f"component of {self.pipe_id} must be Pipe but got {type(self.pipe)}", fl_ctx)
                return
            self.pipe_handler = PipeHandler(
                pipe=self.pipe,
                read_interval=self.read_interval,
                heartbeat_interval=self.heartbeat_interval,
                heartbeat_timeout=self.heartbeat_timeout,
                resend_interval=self.resend_interval,
                max_resends=self.max_resends,
            )
            self.pipe_handler.set_status_cb(self._pipe_status_cb)
            self.pipe.open(self.pipe_channel_name)
            self.pipe_handler.start()
        elif event_type == EventType.END_RUN:
            self.log_info(fl_ctx, "Stopping pipe handler")
            if self.pipe_handler:
                self.pipe_handler.notify_end("end_of_job")
                self.pipe_handler.stop()

    def _pipe_status_cb(self, msg: Message):
        self.logger.info(f"pipe status changed to {msg.topic}")
        self.pipe_handler.stop()

    @abstractmethod
    def shareable_to_exchange_object(self, task_name, task_id, shareable, fl_ctx: FLContext) -> Any:
        pass

    @abstractmethod
    def exchange_object_to_shareable(self, task_name, task_id, data: Any, fl_ctx: FLContext) -> Shareable:
        pass

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        task_id = shareable.get_header(key=FLContextKey.TASK_ID)

        try:
            task_obj = self.shareable_to_exchange_object(task_name, task_id, shareable, fl_ctx)
        except Exception as ex:
            self.log_error(fl_ctx, f"Failed to convert task to exchange object: {secure_format_exception(ex)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # send to peer
        self.log_debug(fl_ctx, "sending task to peer ...")
        req = Message.new_request(topic=task_name, data=task_obj, msg_id=task_id)
        start_time = time.time()
        has_been_read = self.pipe_handler.send_to_peer(req, timeout=self.peer_read_timeout, abort_signal=abort_signal)
        if self.peer_read_timeout and not has_been_read:
            self.log_error(
                fl_ctx,
                f"peer does not accept task '{task_name}' in {time.time()-start_time} secs - aborting task!",
            )
            return make_reply(ReturnCode.SERVICE_UNAVAILABLE)

        # wait for result
        self.log_debug(fl_ctx, "Waiting for result from peer")
        start = time.time()
        while True:
            if abort_signal.triggered:
                # notify peer that the task is aborted
                self.log_debug(fl_ctx, "abort signal triggered!")
                self.pipe_handler.notify_abort(task_id)
                self.pipe_handler.stop()
                return make_reply(ReturnCode.TASK_ABORTED)

            if self.pipe_handler.asked_to_stop:
                self.log_debug(fl_ctx, "task pipe stopped!")
                return make_reply(ReturnCode.TASK_ABORTED)

            reply = self.pipe_handler.get_next()
            if not reply:
                if self.task_wait_time and time.time() - start > self.task_wait_time:
                    # timed out
                    self.log_error(fl_ctx, f"task '{task_name}' timeout after {self.task_wait_time} secs")
                    # also tell peer to abort the task
                    self.pipe_handler.notify_abort(task_id)
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            elif reply.msg_type != Message.REPLY:
                self.log_warning(
                    fl_ctx, f"ignored msg '{reply.topic}.{reply.req_id}' when waiting for '{req.topic}.{req.msg_id}'"
                )
            elif req.topic != reply.topic:
                # ignore wrong task name
                self.log_warning(fl_ctx, f"ignored '{reply.topic}' when waiting for '{req.topic}'")
            elif req.msg_id != reply.req_id:
                self.log_warning(fl_ctx, f"ignored '{reply.req_id}' when waiting for '{req.msg_id}'")
            else:
                self.log_info(fl_ctx, f"got result for request '{task_name}' from peer")

                try:
                    result = self.exchange_object_to_shareable(task_name, task_id, reply.data, fl_ctx)
                    current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
                    if current_round:
                        result.set_header(AppConstants.CURRENT_ROUND, current_round)
                    return result
                except Exception as ex:
                    self.log_error(fl_ctx, f"Failed to convert result: {secure_format_exception(ex)}")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            time.sleep(self.result_poll_interval)


class ShareableTaskExchanger(TaskExchanger):
    """
    ShareableTaskExchanger uses Shareable to exchange task data with the peer.
    """

    def __init__(
        self,
        pipe_id: str,
        read_interval=0.1,
        heartbeat_interval=5.0,
        heartbeat_timeout=30.0,
        resend_interval=2.0,
        max_resends=None,
        peer_read_timeout=5.0,
        task_wait_time=None,
        result_poll_interval=0.5,
        pipe_channel_name=PipeChannelName.TASK,
    ):
        """Constructor of ShareableTaskExchanger

        Args:
            pipe_id: component id of pipe
            read_interval: how often to read from pipe
            heartbeat_interval: how often to send heartbeat to peer
            heartbeat_timeout: max amount of time to allow missing heartbeats before treating peer as dead
            resend_interval: how often to resend a message when failing to send
            max_resends: max number of resends. None means no limit
            peer_read_timeout: time to wait for peer to accept sent message
            task_wait_time: how long to wait for a task to complete. None means waiting forever
            result_poll_interval: how often to poll task result
            pipe_channel_name: the channel name for sending task requests
        """

        super().__init__(
            pipe_id=pipe_id,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            resend_interval=resend_interval,
            max_resends=max_resends,
            peer_read_timeout=peer_read_timeout,
            task_wait_time=task_wait_time,
            result_poll_interval=result_poll_interval,
            pipe_channel_name=pipe_channel_name,
        )

    def shareable_to_exchange_object(self, task_name, task_id, shareable, fl_ctx: FLContext) -> Any:
        return shareable

    def exchange_object_to_shareable(self, task_name, task_id, data: Any, fl_ctx: FLContext) -> Shareable:
        if not isinstance(data, Shareable):
            self.log_error(fl_ctx, f"bad task result from peer: expect Shareable but got {type(data)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        return data


class DXOTaskExchanger(TaskExchanger):
    """
    DXOTaskExchanger uses DXO to exchange task data with the peer.
    """

    def __init__(
        self,
        pipe_id: str,
        read_interval=0.1,
        heartbeat_interval=5.0,
        heartbeat_timeout=30.0,
        resend_interval=2.0,
        max_resends=None,
        peer_read_timeout=5.0,
        task_wait_time=None,
        result_poll_interval=0.5,
        pipe_channel_name=PipeChannelName.TASK,
    ):
        """Constructor of DXOTaskExchanger

        Args:
            pipe_id: component id of pipe
            read_interval: how often to read from pipe
            heartbeat_interval: how often to send heartbeat to peer
            heartbeat_timeout: max amount of time to allow missing heartbeats before treating peer as dead
            resend_interval: how often to resend a message when failing to send
            max_resends: max number of resends. None means no limit
            peer_read_timeout: time to wait for peer to accept sent message
            task_wait_time: how long to wait for a task to complete. None means waiting forever
            result_poll_interval: how often to poll task result
            pipe_channel_name: the channel name for sending task requests
        """
        super().__init__(
            pipe_id=pipe_id,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            resend_interval=resend_interval,
            max_resends=max_resends,
            peer_read_timeout=peer_read_timeout,
            task_wait_time=task_wait_time,
            result_poll_interval=result_poll_interval,
            pipe_channel_name=pipe_channel_name,
        )

    def shareable_to_exchange_object(self, task_name, task_id, shareable, fl_ctx: FLContext) -> Any:
        dxo = from_shareable(shareable)
        ex_task = ExchangeTask(task_name=task_name, task_id=task_id, data=dxo)
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND, None)
        if current_round is not None:
            dxo.set_meta_prop(MetaKey.CURRENT_ROUND, current_round)

        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS, None)
        if total_rounds is not None:
            dxo.set_meta_prop(MetaKey.TOTAL_ROUNDS, total_rounds)
        return ex_task

    def exchange_object_to_shareable(self, task_name, task_id, data: Any, fl_ctx: FLContext) -> Shareable:
        if not isinstance(data, ExchangeTask):
            self.logger.error(f"bad result data from peer - must be ExchangeTask but got {type(data)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        ex_task = data
        if ex_task.return_code:
            rc = ex_task.return_code.lower()
        else:
            rc = "ok"
        if rc != "ok":
            return make_reply(rc)

        if not isinstance(ex_task.data, DXO):
            self.logger.error(f"bad result data from peer - task data must be DXO but got {type(ex_task.data)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        return ex_task.data.to_shareable()
