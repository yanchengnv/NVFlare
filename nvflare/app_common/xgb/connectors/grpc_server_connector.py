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
import nvflare.app_common.xgb.proto.federated_pb2 as pb2
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.xgb.connectors.xgb_connector import XGBServerConnector
from nvflare.app_common.xgb.defs import Constant
from nvflare.app_common.xgb.grpc_client import GrpcClient
from nvflare.fuel.f3.drivers.net_utils import get_open_tcp_port


class GrpcServerConnector(XGBServerConnector):
    def __init__(
        self,
        int_client_grpc_options=None,
        xgb_server_ready_timeout=Constant.XGB_SERVER_READY_TIMEOUT,
        in_process=True,
    ):
        XGBServerConnector.__init__(self, in_process)
        self.int_client_grpc_options = int_client_grpc_options
        self.xgb_server_ready_timeout = xgb_server_ready_timeout
        self.in_process = in_process
        self.internal_xgb_client = None
        self._server_stopped = False
        self._exit_code = 0

    def _start_server(self, addr: str, port: int, world_size: int, fl_ctx: FLContext):
        app_ctx = {
            Constant.APP_CTX_SERVER_ADDR: addr,
            Constant.APP_CTX_WORLD_SIZE: world_size,
            Constant.APP_CTX_PORT: port,
        }
        self.start_applet(app_ctx, fl_ctx)

    def _stop_server(self):
        self._server_stopped = True
        self.stop_applet()

    def _is_stopped(self) -> (bool, int):
        runner_stopped, ec = self.is_applet_stopped()
        if runner_stopped:
            self.logger.info("applet is stopped!")
            return runner_stopped, ec

        if self._server_stopped:
            self.logger.info("grpc server is stopped!")
            return True, self._exit_code

        return False, 0

    def start(self, fl_ctx: FLContext):
        # we dynamically create server address on localhost
        port = get_open_tcp_port(resources={})
        if not port:
            raise RuntimeError("failed to get a port for XGB server")

        server_addr = f"127.0.0.1:{port}"
        self.log_info(fl_ctx, f"starting grpc connector: {server_addr=} {port=} {self.world_size=}")
        self._start_server(server_addr, port, self.world_size, fl_ctx)

        # start XGB client
        self.internal_xgb_client = GrpcClient(server_addr, self.int_client_grpc_options)
        self.internal_xgb_client.start(ready_timeout=self.xgb_server_ready_timeout)

    def stop(self, fl_ctx: FLContext):
        client = self.internal_xgb_client
        self.internal_xgb_client = None
        if client:
            self.log_info(fl_ctx, "Stopping internal XGB client")
            client.stop()
        self._stop_server()

    def all_gather(self, rank: int, seq: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        result = self.internal_xgb_client.send_allgather(seq_num=seq, rank=rank, data=send_buf)
        if isinstance(result, pb2.AllgatherReply):
            return result.receive_buffer
        else:
            raise RuntimeError(f"bad result from XGB server: expect AllgatherReply but got {type(result)}")

    def all_gather_v(self, rank: int, seq: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        result = self.internal_xgb_client.send_allgatherv(seq_num=seq, rank=rank, data=send_buf)
        if isinstance(result, pb2.AllgatherVReply):
            return result.receive_buffer
        else:
            raise RuntimeError(f"bad result from XGB server: expect AllgatherVReply but got {type(result)}")

    def all_reduce(
        self,
        rank: int,
        seq: int,
        data_type: int,
        reduce_op: int,
        send_buf: bytes,
        fl_ctx: FLContext,
    ) -> bytes:
        result = self.internal_xgb_client.send_allreduce(
            seq_num=seq,
            rank=rank,
            data=send_buf,
            data_type=data_type,
            reduce_op=reduce_op,
        )
        if isinstance(result, pb2.AllreduceReply):
            return result.receive_buffer
        else:
            raise RuntimeError(f"bad result from XGB server: expect AllreduceReply but got {type(result)}")

    def broadcast(self, rank: int, seq: int, root: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        result = self.internal_xgb_client.send_broadcast(seq_num=seq, rank=rank, data=send_buf, root=root)
        if isinstance(result, pb2.BroadcastReply):
            return result.receive_buffer
        else:
            raise RuntimeError(f"bad result from XGB server: expect BroadcastReply but got {type(result)}")
