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
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.constants import Mode

from .pipe import Message, Pipe

SSL_ROOT_CERT = "rootCA.pem"
CHANNEL = "ipc_pipe"
TOPIC = "ipc_pipe"


class IPCPipe(Pipe, ABC):
    def __init__(self, mode: Mode):
        super().__init__(mode)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cell = None
        self.peer_fqcn = None
        self.agent_id = None
        self.site_name = None
        self.received_msgs = queue.Queue()
        self.topic = None

    def set_agent_id(self, agent_id):
        self.agent_id = agent_id
        self.topic = f"{TOPIC}_{self.agent_id}"

    def set_cell(self, cell: Cell):
        self.cell = cell
        self.cell.register_request_cb(channel=CHANNEL, topic=self.topic, cb=self._receive_message)
        self.logger.info(f"registered task CB for {CHANNEL} {self.topic}")

    def send(self, msg: Message, timeout=None) -> bool:
        if not self.peer_fqcn:
            # the A-side does not know its peer FQCN until a message is received from the peer (P-side).
            self.logger.warning("peer FQCN is not known yet")
            return False

        reply = self.cell.send_request(
            channel=CHANNEL,
            topic=self.topic,
            target=self.peer_fqcn,
            request=CellMessage(payload=msg),
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
        if not self.peer_fqcn:
            # this is A-side
            self.peer_fqcn = sender
        elif self.peer_fqcn != sender:
            raise RuntimeError(f"peer FQCN mismatch: expect {self.peer_fqcn} but got {sender}")
        self.received_msgs.put_nowait(request.payload)
        return make_reply(ReturnCode.OK)

    def receive(self, timeout=None) -> Union[None, Message]:
        if timeout:
            return self.received_msgs.get(block=True, timeout=timeout)
        else:
            return self.received_msgs.get_nowait()

    def clear(self):
        while not self.received_msgs.empty():
            self.received_msgs.get_nowait()

    @staticmethod
    def agent_cell_name(site_name, name):
        return f"{site_name}--{name}"


class IPCAPipe(IPCPipe):
    """
    The APipe (Active Pipe) is used on the 3rd-party side
    """

    def __init__(
        self,
        site_name: str,
        root_url: str = "",
        secure_mode=True,
        workspace_dir: str = "",
    ):
        super().__init__(Mode.ACTIVE)
        self.root_url = root_url
        self.secure_mode = secure_mode
        self.workspace_dir = workspace_dir
        self.site_name = site_name
        self.net_agent = None
        common_decomposers.register()
        ConfigService.initialize(section_files={}, config_path=[workspace_dir])

    def _build_cell(self, name):
        cell_name = self.agent_cell_name(self.site_name, name)
        credentials = {}
        if self.secure_mode:
            root_cert_path = ConfigService.find_file(SSL_ROOT_CERT)
            if not root_cert_path:
                raise ValueError(f"cannot find {SSL_ROOT_CERT} from config path {self.workspace_dir}")

            credentials = {
                DriverParams.CA_CERT.value: root_cert_path,
            }

        cell = Cell(
            fqcn=cell_name,
            root_url=self.root_url,
            secure=self.secure_mode,
            credentials=credentials,
            create_internal_listener=False,
        )
        self.net_agent = NetAgent(cell)
        self.set_cell(cell)

    def open(self, name: str):
        self.set_agent_id(agent_id=name)
        self._build_cell(name)
        self.cell.start()

    def close(self):
        self.cell.stop()
        self.net_agent.close()


class IPCPPipe(IPCPipe, FLComponent):
    """
    The PPipe (Passive Pipe) is used on FLARE client side.
    """

    def __init__(self):
        super().__init__(Mode.PASSIVE)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            self.set_cell(engine.get_cell())
            self.site_name = fl_ctx.get_identity_name()
            if self.agent_id:
                self.peer_fqcn = self.agent_cell_name(self.site_name, self.agent_id)

    def open(self, name: str):
        self.set_agent_id(agent_id=name)
        if self.site_name:
            self.peer_fqcn = self.agent_cell_name(self.site_name, name)

    def close(self):
        # Passive pipe (on FLARE Client) doesn't need to close anything
        pass
