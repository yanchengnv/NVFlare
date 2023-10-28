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
import queue
import threading
from abc import ABC
from typing import Union

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent, FLContext
from nvflare.app_common.decomposers import common_decomposers
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.cell import Message as CellMessage
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.utils.config_service import search_file
from nvflare.fuel.utils.constants import Mode

from .pipe import Message, Pipe

SSL_ROOT_CERT = "rootCA.pem"
CHANNEL = "ipc_pipe"

_HEADER_MSG_TYPE = "ipc_pipe.msg_type"
_HEADER_MSG_ID = "ipc_pipe.msg_id"
_HEADER_REQ_ID = "ipc_pipe.req_id"


def agent_fqcn(site_name, agent_id):
    return f"{site_name}--{agent_id}"


def to_cell_message(msg: Message) -> CellMessage:
    headers = {
        _HEADER_MSG_TYPE: msg.msg_type,
        _HEADER_MSG_ID: msg.msg_id
    }
    if msg.req_id:
        headers[_HEADER_REQ_ID] = msg.req_id

    return CellMessage(
        headers=headers,
        payload=msg.data
    )


def from_cell_message(cm: CellMessage) -> Message:
    return Message(
        msg_id=cm.get_header(_HEADER_MSG_ID),
        msg_type=cm.get_header(_HEADER_MSG_TYPE),
        topic=cm.get_header(MessageHeaderKey.TOPIC),
        req_id=cm.get_header(_HEADER_REQ_ID),
        data=cm.payload,
    )


class IPCPipe(Pipe, ABC):
    def __init__(self, mode: Mode, agent_id: str):
        """The constructor of the base class.

        Args:
            mode: passive or active mode
            agent_id: unique id of the agent that represents 3rd-party system in cellnet.
        """
        super().__init__(mode)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cell = None
        self.peer_fqcn = None
        self.agent_id = agent_id
        self.site_name = None
        self.received_msgs = queue.Queue()  # contains Message(s), not CellMessage(s)!
        self.channel = None

    def set_cell_cb(self, pipe_name: str):
        # Each pipe must have a unique name. The channel is named after the pipe name.
        # This allows multiple pipes between the two peers (e.g. one pipe for tasks, another for metrics).
        self.channel = f"{CHANNEL}__{pipe_name}"
        self.cell.register_request_cb(channel=self.channel, topic="*", cb=self._receive_message)
        self.logger.info(f"registered request CB for {self.channel}")

    def send(self, msg: Message, timeout=None) -> bool:
        if not self.peer_fqcn:
            # the A-side does not know its peer FQCN until a message is received from the peer (P-side).
            self.logger.warning("peer FQCN is not known yet")
            return False

        reply = self.cell.send_request(
            channel=self.channel,
            topic=msg.topic,
            target=self.peer_fqcn,
            request=to_cell_message(msg),
            timeout=timeout,
        )
        if reply:
            rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
            if rc == ReturnCode.OK:
                return True
            else:
                self.logger.error(f"return code from peer {self.peer_fqcn}: {rc}")
                return False
        else:
            return False

    def _receive_message(self, request: CellMessage) -> Union[None, CellMessage]:
        sender = request.get_header(MessageHeaderKey.ORIGIN)
        topic = request.get_header(MessageHeaderKey.TOPIC)
        self.logger.info(f"got msg from peer {sender}: {topic}")

        if self.mode == Mode.ACTIVE:
            # this is A-side
            self.peer_fqcn = sender
        elif self.peer_fqcn != sender:
            raise RuntimeError(f"peer FQCN mismatch: expect {self.peer_fqcn} but got {sender}")
        msg = from_cell_message(request)
        self.received_msgs.put_nowait(msg)
        return make_reply(ReturnCode.OK)

    def receive(self, timeout=None) -> Union[None, Message]:
        try:
            if timeout:
                return self.received_msgs.get(block=True, timeout=timeout)
            else:
                return self.received_msgs.get_nowait()
        except queue.Empty:
            return None

    def clear(self):
        while not self.received_msgs.empty():
            self.received_msgs.get_nowait()

    def can_resend(self) -> bool:
        return True


class _CellInfo:

    def __init__(self, cell, net_agent):
        self.cell = cell
        self.net_agent = net_agent
        self.started = False
        self.pipes = []
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if not self.started:
                self.cell.start()
                self.started = True

    def add_pipe(self, p):
        with self.lock:
            self.pipes.append(p)

    def close_pipe(self, p):
        with self.lock:
            try:
                self.pipes.remove(p)
                if len(self.pipes) == 0:
                    # all pipes are closed - close cell and agent
                    self.cell.stop()
                    self.net_agent.close()
            except:
                pass


class APipe(IPCPipe, FLComponent):
    """
    The APipe (Active Pipe) is used on the 3rd-party side
    """

    _lock = threading.Lock()
    _cells_info = {}   # root_url => {} of cell_name => _CellInfo
    _initialized = False

    @classmethod
    def _initialize(cls):
        with cls._lock:
            if not cls._initialized:
                cls._initialized = True
                common_decomposers.register()

    @classmethod
    def _build_cell(cls, root_url, site_name, agent_id, secure_mode, workspace_dir):
        """Build a cell if necessary.
        The combination of (root_url, site_name, agent_id) uniquely determine one cell.
        There can be multiple pipes on the same cell.

        Args:
            root_url:
            site_name:
            agent_id:
            secure_mode:
            workspace_dir:

        Returns:

        """
        with cls._lock:
            cell_key = f"{root_url}.{site_name}.{agent_id}"
            cell_name = agent_fqcn(site_name, agent_id)
            ci = cls._cells_info.get(cell_key)
            if not ci:
                credentials = {}
                if secure_mode:
                    root_cert_path = search_file(SSL_ROOT_CERT, workspace_dir)
                    if not root_cert_path:
                        raise ValueError(f"cannot find {SSL_ROOT_CERT} from config path {workspace_dir}")

                    credentials = {
                        DriverParams.CA_CERT.value: root_cert_path,
                    }

                cell = Cell(
                    fqcn=cell_name,
                    root_url=root_url,
                    secure=secure_mode,
                    credentials=credentials,
                    create_internal_listener=False,
                )
                net_agent = NetAgent(cell)
                ci = _CellInfo(cell, net_agent)
                cls._cells_info[cell_key] = ci
            return ci

    def __init__(
        self,
        site_name: str,
        agent_id: str,
        root_url: str = "",
        secure_mode=True,
        workspace_dir: str = "",
    ):
        """Constructor of IPCAPipe.

        Args:
            site_name: name of the FLARE client to communicate with
            agent_id: the unique name of the agent that represents the 3rd-party system in the cellnet
            root_url: root url of the cellnet
            secure_mode: whether connection to the cellnet is secure mode
            workspace_dir: directory of the workspace that contains startup kit
        """
        super().__init__(Mode.ACTIVE, agent_id)
        self.root_url = root_url
        self.secure_mode = secure_mode
        self.workspace_dir = workspace_dir
        self.site_name = site_name
        self.agent_id = agent_id
        self._initialize()
        if agent_id:
            # the hard-coded agent_id
            self.ci = self._build_cell(root_url, site_name, agent_id, secure_mode, workspace_dir)
            self.cell = self.ci.cell
            self.ci.add_pipe(self)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        # self.logger.info(f"APipe handling event {event_type}")
        if event_type == EventType.ABOUT_TO_START_RUN:
            if not self.site_name:
                self.site_name = fl_ctx.get_identity_name()

            if not self.agent_id:
                self.agent_id = fl_ctx.get_job_id()
                self.logger.info(f"building cell {self.site_name=} {self.agent_id=}")
                self.ci = self._build_cell(self.root_url, self.site_name, self.agent_id,
                                           self.secure_mode, self.workspace_dir)
                self.cell = self.ci.cell
                self.ci.add_pipe(self)

    def open(self, name: str):
        self.ci.start()
        self.set_cell_cb(name)

    def close(self):
        self.ci.close_pipe(self)


class PPipe(IPCPipe, FLComponent):
    """
    The PPipe (Passive Pipe) is used on FLARE client side.
    """

    def __init__(self, agent_id: str):
        super().__init__(Mode.PASSIVE, agent_id)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABOUT_TO_START_RUN:
            engine = fl_ctx.get_engine()
            self.cell = engine.get_cell()
            self.site_name = fl_ctx.get_identity_name()
            if not self.agent_id:
                self.agent_id = fl_ctx.get_job_id()
            self.peer_fqcn = agent_fqcn(self.site_name, self.agent_id)

    def open(self, name: str):
        self.set_cell_cb(name)

    def close(self):
        # Passive pipe (on FLARE Client) doesn't need to close anything
        pass
