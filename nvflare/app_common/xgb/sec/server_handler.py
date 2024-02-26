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
import os
import threading

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.xgb.defs import Constant


class ServerSecurityHandler(FLComponent):
    def __init__(self):
        FLComponent.__init__(self)
        self.encrypted_gh = None
        self.gh_source_rank = 0
        self.gh_seq = 0
        self.gh_original_buf_size = 0
        self.aggr_seq = 0
        self.aggr_result_dict = None
        self.aggr_result_list = None
        self.aggr_result_lock = threading.Lock()

    def _handle_before_broadcast(self, fl_ctx: FLContext):
        request = fl_ctx.get_prop(Constant.PARAM_KEY_REQUEST)
        assert isinstance(request, Shareable)
        has_encrypted_gh = request.get_header(Constant.HEADER_KEY_ENCRYPTED_DATA)
        if not has_encrypted_gh:
            return
        self.encrypted_gh = fl_ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)
        self.gh_source_rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK)
        self.gh_seq = fl_ctx.get_prop(Constant.PARAM_KEY_SEQ)
        self.gh_original_buf_size = request.get_header(Constant.HEADER_KEY_ORIGINAL_BUF_SIZE)

        # only need to send a small dummy buffer to the server
        dummy_buf = os.urandom(Constant.DUMMY_BUFFER_SIZE)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEND_BUF, value=dummy_buf, private=True, sticky=False)

    def _handle_after_broadcast(self, fl_ctx: FLContext):
        # this is called after the Server already received broadcast calls from all clients of the same sequence
        rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK)
        seq = fl_ctx.get_prop(Constant.PARAM_KEY_SEQ)
        if seq != self.gh_seq:
            # this is not a gh broadcast
            return

        reply = fl_ctx.get_prop(Constant.PARAM_KEY_REPLY)
        assert isinstance(reply, Shareable)
        reply.set_header(Constant.HEADER_KEY_ENCRYPTED_DATA, True)
        reply.set_header(Constant.HEADER_KEY_ORIGINAL_BUF_SIZE, self.gh_original_buf_size)

        if rank == self.gh_source_rank:
            # no need to send any data back to label client
            fl_ctx.set_prop(key=Constant.PARAM_KEY_SEND_BUF, value=None, private=True, sticky=False)
            return

        # send encrypted ghs
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEND_BUF, value=self.encrypted_gh, private=True, sticky=False)

    def _handle_before_all_gather_v(self, fl_ctx: FLContext):
        request = fl_ctx.get_prop(Constant.PARAM_KEY_REQUEST)
        assert isinstance(request, Shareable)
        has_encrypted_data = request.get_header(Constant.HEADER_KEY_ENCRYPTED_DATA)
        if not has_encrypted_data:
            return

        fl_ctx.set_prop(key="in_aggr", value=True, private=True, sticky=False)
        rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK)

        # the send_buf contains encoded aggr result (str) from this rank
        send_buf = fl_ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)
        with self.aggr_result_lock:
            self.aggr_result_list = None
            if not self.aggr_result_dict:
                self.aggr_result_dict = {}
            self.aggr_result_dict[rank] = send_buf

        # only send a dummy to the Server
        fl_ctx.set_prop(
            key=Constant.PARAM_KEY_SEND_BUF, value=os.urandom(Constant.DUMMY_BUFFER_SIZE), private=True, sticky=False
        )

    def _handle_after_all_gather_v(self, fl_ctx: FLContext):
        # this is called after the Server has finished gathering
        # Note: this fl_ctx is the same as the one in _handle_before_all_gather_v!
        in_aggr = fl_ctx.get_prop("in_aggr")
        if not in_aggr:
            return

        rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK)
        reply = fl_ctx.get_prop(Constant.PARAM_KEY_REPLY)
        assert isinstance(reply, Shareable)
        reply.set_header(Constant.HEADER_KEY_ENCRYPTED_DATA, True)
        with self.aggr_result_lock:
            if not self.aggr_result_list:
                if not self.aggr_result_dict:
                    return self.system_panic(f"Rank {rank}: no aggr result after AllGatherV!", fl_ctx)
                self.aggr_result_list = [None] * len(self.aggr_result_dict)
                for r, v in self.aggr_result_dict.items():
                    self.aggr_result_list[r] = v

                # reset aggr_result_dict for next gather
                self.aggr_result_dict = None
        fl_ctx.set_prop(key=Constant.PARAM_KEY_RCV_BUF, value=self.aggr_result_list, private=True, sticky=False)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == Constant.EVENT_BEFORE_BROADCAST:
            self._handle_before_broadcast(fl_ctx)
        elif event_type == Constant.EVENT_AFTER_BROADCAST:
            self._handle_after_broadcast(fl_ctx)
        elif event_type == Constant.EVENT_BEFORE_ALL_GATHER_V:
            self._handle_before_all_gather_v(fl_ctx)
        elif event_type == Constant.EVENT_AFTER_ALL_GATHER_V:
            self._handle_after_all_gather_v(fl_ctx)
