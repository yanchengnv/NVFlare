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
from typing import Union

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
_PREFIX = "cell_pipe."

_HEADER_MSG_TYPE = _PREFIX + "msg_type"
_HEADER_MSG_ID = _PREFIX + "msg_id"
_HEADER_REQ_ID = _PREFIX + "req_id"


def cell_fqcn(mode, site_name, pipe_id):
    return f"{site_name}--{pipe_id}_{mode}"


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


class CellPipe(Pipe):

    _lock = threading.Lock()
    _cells_info = {}  # root_url => {} of cell_name => _CellInfo
    _initialized = False

    @classmethod
    def _initialize(cls):
        with cls._lock:
            if not cls._initialized:
                cls._initialized = True
                common_decomposers.register()

    @classmethod
    def _build_cell(cls, mode, root_url, site_name, pipe_id, secure_mode, workspace_dir):
        """Build a cell if necessary.
        The combination of (root_url, site_name, pipe_id) uniquely determine one cell.
        There can be multiple pipes on the same cell.

        Args:
            root_url:
            mode:
            site_name:
            pipe_id:
            secure_mode:
            workspace_dir:

        Returns:

        """
        with cls._lock:
            cell_key = f"{root_url}.{site_name}.{pipe_id}"
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
                    fqcn=cell_fqcn(mode, site_name, pipe_id),
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
            mode: Mode,
            site_name: str,
            pipe_id: str,
            root_url: str = "",
            secure_mode=True,
            workspace_dir: str = "",
    ):
        """The constructor of the CellPipe.

        Args:
            mode: passive or active mode
            site_name: name of the FLARE site
            pipe_id: unique id of the pipe.
            root_url: the root url of the cellnet that the pipe's cell will join
            secure_mode: whether connection to the root is secure (TLS)
            workspace_dir: the directory that contains startup kit for joining the cellnet
        """
        super().__init__(mode)
        self._initialize()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.root_url = root_url
        self.secure_mode = secure_mode
        self.workspace_dir = workspace_dir
        self.site_name = site_name
        self.pipe_id = pipe_id

        mode = str(mode).strip().lower()
        self.ci = self._build_cell(mode, root_url, site_name, pipe_id, secure_mode, workspace_dir)
        self.cell = self.ci.cell
        self.ci.add_pipe(self)

        if mode == "active":
            peer_mode = "passive"
        else:
            peer_mode = "active"
        self.peer_fqcn = cell_fqcn(peer_mode, site_name, pipe_id)
        self.received_msgs = queue.Queue()  # contains Message(s), not CellMessage(s)!
        self.channel = None
        self.logger.info(f"My Mode {mode}. Peer Cell {self.peer_fqcn}")

    def set_cell_cb(self, channel_name: str):
        # This allows multiple channels over the same pipe (e.g. one channel for tasks, another for metrics).
        self.channel = f"{_PREFIX}__{channel_name}"
        self.cell.register_request_cb(channel=self.channel, topic="*", cb=self._receive_message)
        self.logger.info(f"registered request CB for {self.channel}")

    def send(self, msg: Message, timeout=None) -> bool:
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
        self.logger.debug(f"got msg from peer {sender}: {topic}")

        if self.peer_fqcn != sender:
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

    def open(self, name: str):
        self.ci.start()
        self.set_cell_cb(name)

    def close(self):
        self.ci.close_pipe(self)
