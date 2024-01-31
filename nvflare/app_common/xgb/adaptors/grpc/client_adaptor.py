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

import nvflare.app_common.xgb.adaptors.grpc.proto.federated_pb2 as pb2
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.xgb.adaptor import XGBClientAdaptor
from nvflare.app_common.xgb.adaptors.grpc.proto.federated_pb2_grpc import FederatedServicer
from nvflare.app_common.xgb.adaptors.grpc.server import XGBServer
from nvflare.fuel.f3.drivers.net_utils import get_open_tcp_port
from nvflare.security.logging import secure_format_exception


class GrpcClientAdaptor(XGBClientAdaptor, FederatedServicer):
    def __init__(
        self,
        grpc_options=None,
        req_timeout=10.0,
    ):
        XGBClientAdaptor.__init__(self, req_timeout)
        self.grpc_options = grpc_options
        self.internal_xgb_server = None
        self.stopped = False
        self.internal_server_addr = None

    def start_client(self, server_addr: str, port: int):
        pass

    def stop_client(self):
        pass

    def is_client_stopped(self) -> (bool, int):
        pass

    def start(self, fl_ctx: FLContext):
        if self.rank is None:
            raise RuntimeError("cannot start - my rank is not set")

        if not self.num_rounds:
            raise RuntimeError("cannot start - num_rounds is not set")

        # dynamically determine address on localhost
        port = get_open_tcp_port(resources={})
        if not port:
            raise RuntimeError("failed to get a port for XGB server")
        self.internal_server_addr = f"127.0.0.1:{port}"
        self.logger.info(f"Start internal server at {self.internal_server_addr}")
        self.internal_xgb_server = XGBServer(self.internal_server_addr, 10, self.grpc_options, self)
        self.internal_xgb_server.start(no_blocking=True)
        self.logger.info(f"Started internal server at {self.internal_server_addr}")
        self.start_client(self.internal_server_addr, port)
        self.logger.info(f"Started external XGB Client")

    def stop(self, fl_ctx: FLContext):
        if self.stopped:
            return

        self.stopped = True
        self.stop_client()

        if self.internal_xgb_server:
            self.logger.info("Stop internal XGB Server")
            self.internal_xgb_server.shutdown()

    def _is_stopped(self) -> (bool, int):
        return self.is_client_stopped()

    def _abort(self, reason: str):
        # stop the gRPC XGB client (the target)
        self.abort_signal.trigger(True)

        # abort the FL client
        with self.engine.new_context() as fl_ctx:
            self.system_panic(reason, fl_ctx)

    def Allgather(self, request: pb2.AllgatherRequest, context):
        try:
            rcv_buf = self._send_all_gather(
                rank=request.rank,
                seq=request.sequence_number,
                send_buf=request.send_buffer,
            )
            return pb2.AllgatherReply(receive_buffer=rcv_buf)
        except Exception as ex:
            self._abort(reason=f"send_all_gather exception: {secure_format_exception(ex)}")
            return None

    def AllgatherV(self, request: pb2.AllgatherVRequest, context):
        try:
            rcv_buf = self._send_all_gather_v(
                rank=request.rank,
                seq=request.sequence_number,
                send_buf=request.send_buffer,
            )
            return pb2.AllgatherVReply(receive_buffer=rcv_buf)
        except Exception as ex:
            self._abort(reason=f"send_all_gather_v exception: {secure_format_exception(ex)}")
            return None

    def Allreduce(self, request: pb2.AllreduceRequest, context):
        try:
            rcv_buf = self._send_all_reduce(
                rank=request.rank,
                seq=request.sequence_number,
                data_type=request.data_type,
                reduce_op=request.reduce_operation,
                send_buf=request.send_buffer,
            )
            return pb2.AllreduceReply(receive_buffer=rcv_buf)
        except Exception as ex:
            self._abort(reason=f"send_all_reduce exception: {secure_format_exception(ex)}")
            return None

    def Broadcast(self, request: pb2.BroadcastRequest, context):
        try:
            rcv_buf = self._send_broadcast(
                rank=request.rank,
                seq=request.sequence_number,
                root=request.root,
                send_buf=request.send_buffer,
            )
            return pb2.BroadcastReply(receive_buffer=rcv_buf)
        except Exception as ex:
            self._abort(reason=f"send_broadcast exception: {secure_format_exception(ex)}")
            return None
