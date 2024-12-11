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
import concurrent.futures
import logging
import threading
import time
import uuid

from nvflare.apis.aux_spec import AuxMessenger
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ConfigVarName, SystemConfigs
from nvflare.apis.fl_context import FLContext
from nvflare.apis.rm import PROP_KEY_DEBUG_INFO
from nvflare.apis.shareable import ReservedHeaderKey, ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils.fl_context_utils import generate_log_message
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.registry import Registry
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.validation_utils import check_callable, check_positive_number, check_str
from nvflare.security.logging import secure_format_exception, secure_format_traceback

# Operation Types
OP_REQUEST = "req"
OP_QUERY = "query"
OP_REPLY = "reply"

# Reliable Message headers
HEADER_OP = "rm.op"
HEADER_CHANNEL = "rm.channel"
HEADER_TOPIC = "rm.topic"
HEADER_TX_ID = "rm.tx_id"
HEADER_PER_MSG_TIMEOUT = "rm.per_msg_timeout"
HEADER_TX_TIMEOUT = "rm.tx_timeout"
HEADER_STATUS = "rm.status"

# Status
STATUS_IN_PROCESS = "in_process"
STATUS_IN_REPLY = "in_reply"
STATUS_NOT_RECEIVED = "not_received"
STATUS_REPLIED = "replied"
STATUS_ABORTED = "aborted"
STATUS_DUP_REQUEST = "dup_request"

# Topics for Reliable Message
TOPIC_RELIABLE_REQUEST = "RM.RELIABLE_REQUEST"
TOPIC_RELIABLE_REPLY = "RM.RELIABLE_REPLY"

PROP_KEY_TX_ID = "RM.TX_ID"
PROP_KEY_CHANNEL = "RM.CHANNEL"
PROP_KEY_TOPIC = "RM.TOPIC"
PROP_KEY_OP = "RM.OP"


def _extract_result(reply: dict, target: str):
    err_rc = ReturnCode.COMMUNICATION_ERROR
    if not isinstance(reply, dict):
        return make_reply(err_rc), err_rc
    result = reply.get(target)
    if not result:
        return make_reply(err_rc), err_rc
    return result, result.get_return_code()


def _status_reply(status: str):
    return make_reply(rc=ReturnCode.OK, headers={HEADER_STATUS: status})


def _error_reply(rc: str, error: str):
    return make_reply(rc, headers={ReservedHeaderKey.ERROR: error})


class _RequestReceiver:
    """This class handles reliable message request on the receiving end"""

    def __init__(self, messenger, channel, topic, handler_info, executor, per_msg_timeout, tx_timeout):
        """The constructor

        Args:
            topic: The topic of the reliable message
            handler_info: The callback function and args to handle the request in the form of
                request_handler_f(channel: str, topic: str, request: Shareable, fl_ctx:FLContext, **kwargs)
            executor: A ThreadPoolExecutor
        """
        self.messenger = messenger
        self.channel = channel
        self.topic = topic
        self.handler_info = handler_info
        self.executor = executor
        self.per_msg_timeout = per_msg_timeout
        self.tx_timeout = tx_timeout
        self.rcv_time = None
        self.result = None
        self.source = None
        self.msg_secure = None
        self.msg_optional = None
        self.tx_id = None
        self.reply_time = None
        self.replying = False
        self.lock = threading.Lock()

    def process(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        with self.lock:
            self.tx_id = request.get_header(HEADER_TX_ID)
            op = request.get_header(HEADER_OP)
            peer_ctx = fl_ctx.get_peer_context()
            assert isinstance(peer_ctx, FLContext)
            self.source = peer_ctx.get_identity_name()
            msg = request.get_cell_message()
            assert isinstance(msg, Message)
            self.msg_secure = msg.get_header(MessageHeaderKey.SECURE, False)
            self.msg_optional = msg.get_header(MessageHeaderKey.OPTIONAL, False)

            if op == OP_REQUEST:
                # it is possible that a new request for the same tx is received while we are processing the previous one
                if not self.rcv_time:
                    self.rcv_time = time.time()
                    self.per_msg_timeout = request.get_header(HEADER_PER_MSG_TIMEOUT)
                    self.tx_timeout = request.get_header(HEADER_TX_TIMEOUT)

                    # start processing
                    self.messenger.info(fl_ctx, f"started processing request of topic {self.topic}")
                    try:
                        self.executor.submit(self._do_request, request, fl_ctx)
                        return _status_reply(STATUS_IN_PROCESS)  # ack
                    except Exception as ex:
                        # it is possible that the RM is already closed (self.executor is shut down)
                        self.messenger.error(fl_ctx, f"failed to submit request: {secure_format_exception(ex)}")
                        return make_reply(ReturnCode.SERVICE_UNAVAILABLE)
                elif self.result:
                    # we already finished processing - send the result back
                    self.messenger.info(fl_ctx, "resend result back to requester")
                    return self.result
                else:
                    # we are still processing
                    self.messenger.info(fl_ctx, "got request - the request is being processed")
                    return _status_reply(STATUS_IN_PROCESS)
            elif op == OP_QUERY:
                if self.result:
                    if self.reply_time:
                        # result already sent back successfully
                        self.messenger.info(fl_ctx, "got query: we already replied successfully")
                        return _status_reply(STATUS_REPLIED)
                    elif self.replying:
                        # result is being sent
                        self.messenger.info(fl_ctx, "got query: reply is being sent")
                        return _status_reply(STATUS_IN_REPLY)
                    else:
                        # try to send the result again
                        self.messenger.info(fl_ctx, "got query: sending reply again")
                        return self.result
                else:
                    # still in process
                    if time.time() - self.rcv_time > self.tx_timeout:
                        # the process is taking too much time
                        self.messenger.error(
                            fl_ctx, f"aborting processing since exceeded max tx time {self.tx_timeout}"
                        )
                        return _status_reply(STATUS_ABORTED)
                    else:
                        self.messenger.debug(fl_ctx, "got query: request is in-process")
                        return _status_reply(STATUS_IN_PROCESS)

    def _try_reply(self, fl_ctx: FLContext):
        self.replying = True
        start_time = time.time()
        self.messenger.debug(fl_ctx, f"try to send reply back to {self.source}: {self.per_msg_timeout=}")
        ack = self.messenger.aux_runner.send_aux_request(
            targets=[self.source],
            topic=TOPIC_RELIABLE_REPLY,
            request=self.result,
            timeout=self.per_msg_timeout,
            fl_ctx=fl_ctx,
            secure=self.msg_secure,
            optional=self.msg_optional,
        )
        time_spent = time.time() - start_time
        self.replying = False
        _, rc = _extract_result(ack, self.source)
        if rc == ReturnCode.OK:
            # reply sent successfully!
            self.reply_time = time.time()
            self.messenger.debug(fl_ctx, f"sent reply successfully in {time_spent} secs")

            # release the receiver kept by the ReliableMessage!
            self.messenger.release_request_receiver(self, fl_ctx)
        else:
            # unsure whether the reply was sent successfully
            # do not release the request receiver in case the requester asks for result in a query
            self.messenger.error(
                fl_ctx, f"failed to send reply in {time_spent} secs: {rc=}; will wait for requester to query"
            )

    def _do_request(self, request: Shareable, fl_ctx: FLContext):
        start_time = time.time()
        self.messenger.debug(fl_ctx, "invoking request handler")
        try:
            handler_f, handler_kwargs = self.handler_info
            result = handler_f(self.channel, self.topic, request, fl_ctx, **handler_kwargs)
        except Exception as e:
            self.messenger.error(fl_ctx, f"exception processing request: {secure_format_traceback()}")
            result = _error_reply(ReturnCode.EXECUTION_EXCEPTION, secure_format_exception(e))

        # send back
        result.set_header(HEADER_TX_ID, self.tx_id)
        result.set_header(HEADER_OP, OP_REPLY)
        result.set_header(HEADER_TOPIC, self.topic)
        result.set_header(HEADER_CHANNEL, self.channel)
        self.result = result
        self.messenger.debug(fl_ctx, f"finished request handler in {time.time() - start_time} secs")
        self._try_reply(fl_ctx)


class _ReplyReceiver:
    """This class handles reliable message replies on the sending end"""

    def __init__(self, tx_id: str, per_msg_timeout: float, tx_timeout: float):
        self.tx_id = tx_id
        self.tx_start_time = time.time()
        self.tx_timeout = tx_timeout
        self.per_msg_timeout = per_msg_timeout
        self.result = None
        self.result_ready = threading.Event()

    def process(self, reply: Shareable) -> Shareable:
        self.result = reply
        self.result_ready.set()
        return make_reply(ReturnCode.OK)


class ReliableMessenger(FLComponent):
    def __init__(self, aux_runner: AuxMessenger):
        FLComponent.__init__(self)
        self.aux_runner = aux_runner
        self.registry = Registry()
        self.req_receivers = {}  # tx id => receiver
        self.req_completed = {}  # tx id => expiration
        self.reply_receivers = {}  # tx id => receiver
        self.tx_lock = threading.Lock()
        self.shutdown_asked = False
        self.logger = get_obj_logger(self)

        aux_runner.register_aux_message_handler(
            topic=TOPIC_RELIABLE_REQUEST,
            message_handle_func=self._receive_request,
        )
        aux_runner.register_aux_message_handler(
            topic=TOPIC_RELIABLE_REPLY,
            message_handle_func=self._receive_reply,
        )

        max_request_workers = ConfigService.get_int_var(
            name=ConfigVarName.RM_MAX_REQUEST_WORKERS, conf=SystemConfigs.APPLICATION_CONF, default=20
        )
        query_interval = ConfigService.get_float_var(
            name=ConfigVarName.RM_QUERY_INTERVAL, conf=SystemConfigs.APPLICATION_CONF, default=2.0
        )

        self.query_interval = query_interval
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_request_workers)
        t = threading.Thread(target=self._monitor_req_receivers, daemon=True)
        t.start()
        self.logger.info(f"started reliable messenger: {max_request_workers=} {query_interval=}")

    def register_request_handler(self, channel: str, topic: str, handler_f, **handler_kwargs):
        """Register a handler for the reliable message with this topic

        Args:
            channel: channel of the reliable message
            topic: The topic of the reliable message
            handler_f: The callback function to handle the request in the form of
                handler_f(topic, request, fl_ctx)
        """
        check_str("channel", channel)
        check_str("topic", topic)
        check_callable("handler_f", handler_f)

        self.registry.set(channel, topic, (handler_f, handler_kwargs))
        self.logger.info(f"registered RM handler: {channel=} {topic=} {handler_f.__name__}")

    def _get_or_create_receiver(self, channel: str, topic: str, request: Shareable, handler_info) -> _RequestReceiver:
        tx_id = request.get_header(HEADER_TX_ID)
        if not tx_id:
            raise RuntimeError("missing tx_id in request")
        with self.tx_lock:
            receiver = self.req_receivers.get(tx_id)
            if not receiver:
                per_msg_timeout = request.get_header(HEADER_PER_MSG_TIMEOUT)
                if not per_msg_timeout:
                    raise RuntimeError("missing per_msg_timeout in request")
                tx_timeout = request.get_header(HEADER_TX_TIMEOUT)
                if not tx_timeout:
                    raise RuntimeError("missing tx_timeout in request")
                receiver = _RequestReceiver(
                    self, channel, topic, handler_info, self.executor, per_msg_timeout, tx_timeout
                )
                self.req_receivers[tx_id] = receiver
            return receiver

    def _receive_request(self, topic: str, request: Shareable, fl_ctx: FLContext):
        tx_id = request.get_header(HEADER_TX_ID)
        op = request.get_header(HEADER_OP)
        rm_channel = request.get_header(HEADER_CHANNEL)
        rm_topic = request.get_header(HEADER_TOPIC)
        fl_ctx.set_prop(key=PROP_KEY_TX_ID, value=tx_id, sticky=False, private=True)
        fl_ctx.set_prop(key=PROP_KEY_OP, value=op, sticky=False, private=True)
        fl_ctx.set_prop(key=PROP_KEY_TOPIC, value=rm_topic, sticky=False, private=True)
        fl_ctx.set_prop(key=PROP_KEY_CHANNEL, value=rm_channel, sticky=False, private=True)
        self.debug(fl_ctx, f"received aux msg ({topic=}) for RM request")

        if op == OP_REQUEST:
            handler_info = self.registry.find(rm_channel, rm_topic)
            if not handler_info:
                # no handler registered for this topic!
                self.error(fl_ctx, "no handler registered for request")
                return make_reply(ReturnCode.TOPIC_UNKNOWN)

            # check whether the request is still standing or completed
            # we should check to get the receiver first, and check req_completed next:
            # if the receiver does not exist in _req_receivers and already completed,
            # then it must exist in _req_completed (since we put it in _req_completed before removing it
            # from _req_receivers).
            receiver = self.req_receivers.get(tx_id)
            if not receiver:
                # no standing process for this request
                # further check whether this request was already completed
                if self.req_completed.get(tx_id):
                    # this request was already completed!
                    self.debug(fl_ctx, "Completed tx_id received")
                    return _status_reply(STATUS_DUP_REQUEST)

            if not receiver:
                # this is a valid new request
                receiver = self._get_or_create_receiver(rm_channel, rm_topic, request, handler_info)

            self.debug(fl_ctx, "received request")
            return receiver.process(request, fl_ctx)
        elif op == OP_QUERY:
            receiver = self.req_receivers.get(tx_id)
            if not receiver:
                # no standing process for this request - is it already completed?
                if self.req_completed.get(tx_id):
                    # the request is already completed
                    return _status_reply(STATUS_REPLIED)

                self.warning(fl_ctx, "received query but the request is not received or already done!")
                return _status_reply(STATUS_NOT_RECEIVED)  # meaning the request wasn't received
            else:
                return receiver.process(request, fl_ctx)
        else:
            self.error(fl_ctx, f"received invalid op {op} for the request")
            return make_reply(rc=ReturnCode.BAD_REQUEST_DATA)

    def _receive_reply(self, topic: str, request: Shareable, fl_ctx: FLContext):
        tx_id = request.get_header(HEADER_TX_ID)
        rm_channel = request.get_header(HEADER_CHANNEL)
        rm_topic = request.get_header(HEADER_TOPIC)
        fl_ctx.set_prop(key=PROP_KEY_TX_ID, value=tx_id, private=True, sticky=False)
        fl_ctx.set_prop(key=PROP_KEY_CHANNEL, value=rm_channel, private=True, sticky=False)
        fl_ctx.set_prop(key=PROP_KEY_TOPIC, value=rm_topic, private=True, sticky=False)
        self.debug(fl_ctx, "received RM reply")
        receiver = self.reply_receivers.get(tx_id)
        if not receiver:
            self.warning(fl_ctx, "received reply but we are no longer waiting for it")
        else:
            assert isinstance(receiver, _ReplyReceiver)
            self.debug(fl_ctx, f"received reply in {time.time() - receiver.tx_start_time} secs - set waiter")
            receiver.process(request)
        return make_reply(ReturnCode.OK)

    def release_request_receiver(self, receiver: _RequestReceiver, fl_ctx: FLContext):
        """Release the specified _RequestReceiver from the receiver table.
        This is to be called after the received request is finished.

        Args:
            receiver: the _RequestReceiver to be released
            fl_ctx: the FL Context

        Returns: None

        """
        with self.tx_lock:
            self._register_completed_req(receiver.tx_id, receiver.tx_timeout)
            self.req_receivers.pop(receiver.tx_id, None)
            self.debug(fl_ctx, f"released request receiver of TX {receiver.tx_id}")

    def _monitor_req_receivers(self):
        while not self.shutdown_asked:
            expired_receivers = []
            with self.tx_lock:
                now = time.time()
                for tx_id, receiver in self.req_receivers.items():
                    assert isinstance(receiver, _RequestReceiver)
                    if receiver.rcv_time and now - receiver.rcv_time > receiver.tx_timeout:
                        self.logger.info(f"detected expired request receiver {tx_id}")
                        expired_receivers.append(tx_id)

            if expired_receivers:
                with self.tx_lock:
                    for tx_id in expired_receivers:
                        self.req_receivers.pop(tx_id, None)

            time.sleep(2.0)
        self.logger.info("shutdown reliable message monitor")

    def shutdown(self):
        """Shutdown ReliableMessage.

        Returns:

        """
        if not self.shutdown_asked:
            self.shutdown_asked = True
            self.executor.shutdown(wait=False)
            self.logger.info("ReliableMessage is shutdown")

    def _log_msg(self, fl_ctx: FLContext, msg: str, level):
        msg = generate_log_message(
            fl_ctx,
            msg,
            ctx_keys={
                PROP_KEY_TX_ID: "rm_tx",
                PROP_KEY_OP: "rm_op",
                PROP_KEY_TOPIC: "rm_topic",
                PROP_KEY_CHANNEL: "rm_channel",
                PROP_KEY_DEBUG_INFO: "debug",
            },
        )
        self.logger.log(level, msg)

    def info(self, fl_ctx: FLContext, msg: str):
        self._log_msg(fl_ctx, msg, level=logging.INFO)

    def warning(self, fl_ctx: FLContext, msg: str):
        self._log_msg(fl_ctx, msg, level=logging.WARNING)

    def error(self, fl_ctx: FLContext, msg: str):
        self._log_msg(fl_ctx, msg, level=logging.ERROR)

    def debug(self, fl_ctx: FLContext, msg: str):
        self._log_msg(fl_ctx, msg, level=logging.DEBUG)

    def send_request(
        self,
        target: str,
        channel: str,
        topic: str,
        request: Shareable,
        per_msg_timeout: float,
        tx_timeout: float,
        fl_ctx: FLContext,
        optional=False,
        secure=False,
    ) -> Shareable:
        """Send a request reliably.

        Args:
            target: The target cell of this request.
            channel: channel of the request
            topic: The topic of the request.
            request: The request to be sent.
            per_msg_timeout (float): Number of seconds to wait for each message before timing out.
            tx_timeout (float): Timeout for the entire transaction.
            fl_ctx (FLContext): Context for federated learning.
            secure: whether to use P2P security for this message
            optional: whether the message is optional

        Returns:
            The reply from the peer.

        Note:
            If `tx_timeout` is not specified or is less than or equal to `per_msg_timeout`,
            the request will be sent only once without retrying.

        """
        check_str("target", target)
        check_positive_number("per_msg_timeout", per_msg_timeout)
        if tx_timeout:
            check_positive_number("tx_timeout", tx_timeout)

        if not tx_timeout or tx_timeout <= per_msg_timeout:
            tx_timeout = per_msg_timeout

        tx_id = str(uuid.uuid4())
        fl_ctx.set_prop(key=PROP_KEY_TX_ID, value=tx_id, private=True, sticky=False)
        fl_ctx.set_prop(key=PROP_KEY_TOPIC, value=topic, private=True, sticky=False)
        fl_ctx.set_prop(key=PROP_KEY_CHANNEL, value=channel, private=True, sticky=False)
        fl_ctx.set_prop(key=PROP_KEY_OP, value=OP_REQUEST, private=True, sticky=False)

        self.info(fl_ctx, f"send request with Reliable Msg {per_msg_timeout=} {tx_timeout=}")
        receiver = _ReplyReceiver(tx_id, per_msg_timeout, tx_timeout)
        self.reply_receivers[tx_id] = receiver
        request.set_header(HEADER_TX_ID, tx_id)
        request.set_header(HEADER_OP, OP_REQUEST)
        request.set_header(HEADER_TOPIC, topic)
        request.set_header(HEADER_PER_MSG_TIMEOUT, per_msg_timeout)
        request.set_header(HEADER_TX_TIMEOUT, tx_timeout)
        try:
            result = self._send_request(target, request, fl_ctx, receiver, secure, optional)
        except Exception as e:
            self.error(fl_ctx, f"exception sending reliable message: {secure_format_traceback()}")
            result = _error_reply(ReturnCode.ERROR, secure_format_exception(e))
        self.reply_receivers.pop(tx_id)
        return result

    def _send_request(
        self,
        target: str,
        request: Shareable,
        fl_ctx: FLContext,
        receiver: _ReplyReceiver,
        secure: bool,
        optional: bool,
    ) -> Shareable:
        # keep sending the request until a positive ack or result is received
        tx_timeout = receiver.tx_timeout
        per_msg_timeout = receiver.per_msg_timeout
        num_tries = 0
        abort_signal = fl_ctx.get_run_abort_signal()
        while True:
            if abort_signal and abort_signal.triggered:
                self.info(fl_ctx, "send_request abort triggered")
                return make_reply(ReturnCode.TASK_ABORTED)

            if time.time() - receiver.tx_start_time >= receiver.tx_timeout:
                self.error(fl_ctx, f"aborting send_request since exceeded {tx_timeout=}")
                return make_reply(ReturnCode.COMMUNICATION_ERROR)

            # it is possible that a reply is already received while we are still trying to send!
            if receiver.result_ready.is_set():
                self.debug(fl_ctx, "result received while in the send loop")
                break

            if num_tries > 0:
                self.debug(fl_ctx, f"retry #{num_tries} sending request: {per_msg_timeout=}")

            ack = self.aux_runner.send_aux_request(
                targets=[target],
                topic=TOPIC_RELIABLE_REQUEST,
                request=request,
                timeout=per_msg_timeout,
                fl_ctx=fl_ctx,
                secure=secure,
                optional=optional,
            )

            # it is possible that a reply is already received while we are waiting for the ack!
            if receiver.result_ready.is_set():
                self.debug(fl_ctx, "result received while waiting for ack")
                break

            ack, rc = _extract_result(ack, target)
            if ack and rc not in [ReturnCode.COMMUNICATION_ERROR]:
                # is this result?
                op = ack.get_header(HEADER_OP)
                if op == OP_REPLY:
                    # the reply is already the result - we are done!
                    # this could happen when we didn't get positive ack for our first request, and the result was
                    # already produced when we did the 2nd request (this request).
                    self.debug(fl_ctx, f"C1: received result in {time.time() - receiver.tx_start_time} seconds; {rc=}")
                    return ack

                # the ack is a status report - check status
                status = ack.get_header(HEADER_STATUS)
                if status and status != STATUS_NOT_RECEIVED:
                    # status should never be STATUS_NOT_RECEIVED, unless there is a bug in the receiving logic
                    # STATUS_NOT_RECEIVED is only possible during "query" phase.
                    self.debug(fl_ctx, f"received status ack: {rc=} {status=}")
                    break

            if time.time() + self.query_interval - receiver.tx_start_time >= tx_timeout:
                self.error(fl_ctx, f"aborting send_request since it will exceed {tx_timeout=}")
                return make_reply(ReturnCode.COMMUNICATION_ERROR)

            # we didn't get a positive ack - wait a short time and re-send the request.
            self.debug(fl_ctx, f"unsure the request was received ({rc=}): will retry in {self.query_interval} secs")
            num_tries += 1
            start = time.time()
            while time.time() - start < self.query_interval:
                if abort_signal and abort_signal.triggered:
                    self.info(fl_ctx, "abort send_request triggered by signal")
                    return make_reply(ReturnCode.TASK_ABORTED)
                time.sleep(0.1)

        self.debug(fl_ctx, "request was received by the peer - will query for result")
        return self._query_result(target, abort_signal, fl_ctx, receiver, secure, optional)

    def _query_result(
        self,
        target: str,
        abort_signal: Signal,
        fl_ctx: FLContext,
        receiver: _ReplyReceiver,
        secure: bool,
        optional: bool,
    ) -> Shareable:
        tx_timeout = receiver.tx_timeout
        per_msg_timeout = receiver.per_msg_timeout

        # Querying phase - try to get result
        query = Shareable()
        query.set_header(HEADER_TX_ID, receiver.tx_id)
        query.set_header(HEADER_OP, OP_QUERY)

        num_tries = 0
        last_query_time = 0
        short_wait = 0.1
        while True:
            if time.time() - receiver.tx_start_time > tx_timeout:
                self.error(fl_ctx, f"aborted query since exceeded {tx_timeout=}")
                return _error_reply(ReturnCode.COMMUNICATION_ERROR, f"max tx timeout ({tx_timeout}) reached")

            if receiver.result_ready.wait(short_wait):
                # we already received result sent by the target.
                # Note that we don't wait forever here - we only wait for _query_interval, so we could
                # check other condition and/or send query to ask for result.
                self.debug(fl_ctx, f"C2: received result in {time.time() - receiver.tx_start_time} seconds")
                return receiver.result

            if abort_signal and abort_signal.triggered:
                self.info(fl_ctx, "aborted query triggered by abort signal")
                return make_reply(ReturnCode.TASK_ABORTED)

            if time.time() - last_query_time < self.query_interval:
                # don't query too quickly
                continue

            # send a query. The ack of the query could be the result itself, or a status report.
            # Note: the ack could be the result because we failed to receive the result sent by the target earlier.
            num_tries += 1
            self.debug(fl_ctx, f"query #{num_tries}: try to get result from {target}: {per_msg_timeout=}")
            ack = self.aux_runner.send_aux_request(
                targets=[target],
                topic=TOPIC_RELIABLE_REQUEST,
                request=query,
                timeout=per_msg_timeout,
                fl_ctx=fl_ctx,
                secure=secure,
                optional=optional,
            )

            # Ignore query result if reply result is already received
            if receiver.result_ready.is_set():
                return receiver.result

            last_query_time = time.time()
            ack, rc = _extract_result(ack, target)
            if ack and rc not in [ReturnCode.COMMUNICATION_ERROR]:
                op = ack.get_header(HEADER_OP)
                if op == OP_REPLY:
                    # the ack is result itself!
                    self.debug(fl_ctx, f"C3: received result in {time.time() - receiver.tx_start_time} seconds")
                    return ack

                status = ack.get_header(HEADER_STATUS)
                if status == STATUS_NOT_RECEIVED:
                    # the receiver side lost context!
                    self.error(fl_ctx, f"peer {target} lost request!")
                    return _error_reply(ReturnCode.EXECUTION_EXCEPTION, "STATUS_NOT_RECEIVED")
                elif status == STATUS_ABORTED:
                    self.error(fl_ctx, f"peer {target} aborted processing!")
                    return _error_reply(ReturnCode.EXECUTION_EXCEPTION, "Aborted")

                self.debug(fl_ctx, f"will retry query in {self.query_interval} secs: {rc=} {status=} {op=}")
            else:
                self.debug(fl_ctx, f"will retry query in {self.query_interval} secs: {rc=}")

    def _register_completed_req(self, tx_id, tx_timeout):
        # Remove expired entries, need to use a copy of the keys
        now = time.time()
        for key in list(self.req_completed.keys()):
            expiration = self.req_completed.get(key)
            if expiration and expiration < now:
                self.req_completed.pop(key, None)

        # Expire in 2 x tx_timeout
        self.req_completed[tx_id] = now + 2 * tx_timeout
