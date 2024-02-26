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
import time

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.xgb.defs import Constant
from nvflare.app_common.xgb.paillier.adder import Adder
from nvflare.app_common.xgb.paillier.decrypter import Decrypter
from nvflare.app_common.xgb.paillier.encryptor import Encryptor
from nvflare.app_common.xgb.paillier.util import (
    combine,
    decode_encrypted_data,
    decode_feature_aggregations,
    encode_encrypted_data,
    encode_feature_aggregations,
    generate_keys,
    split,
)
from nvflare.app_common.xgb.sec.data_converter import DataConverter


class ClientSecurityHandler(FLComponent):
    def __init__(self, key_length=1024, num_workers=10):
        FLComponent.__init__(self)
        self.num_workers = num_workers
        self.key_length = key_length
        self.public_key = None
        self.private_key = None
        self.encryptor = None
        self.adder = None
        self.decrypter = None
        self.data_converter = DataConverter()
        self.encrypted_ghs = None
        self.clear_ghs = None  # for label client: tuple of (g_list, h_list)
        self.original_gh_buffer = None
        self.feature_masks = None

    def _abort(self, error: str, fl_ctx: FLContext):
        self.log_error(fl_ctx, error)
        self.system_panic(reason=error, fl_ctx=fl_ctx)

    def _process_before_broadcast(self, fl_ctx: FLContext):
        root = fl_ctx.get_prop(Constant.PARAM_KEY_ROOT)
        rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK)
        if root != rank:
            # I am not the source of the broadcast
            return

        buffer = fl_ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)
        clear_ghs = self.data_converter.decode_gh_pairs(buffer)
        if clear_ghs is None:
            # the buffer does not contain (g, h) pairs
            return

        self.clear_ghs = clear_ghs
        self.original_gh_buffer = buffer

        # encrypt clear-text gh pairs and send to server
        g_values, h_values = clear_ghs
        gh_combined = [combine(g_values[i], h_values[i]) for i in range(len(g_values))]
        t = time.time()
        encrypted_values = self.encryptor.encrypt(gh_combined)
        print(f"encrypted items: {len(encrypted_values)}, took {time.time() - t} secs")

        t = time.time()
        encoded = encode_encrypted_data(self.public_key, encrypted_values)
        print(f"encoded msg: size={len(encoded)} bytes, time={time.time() - t} secs")

        # Remember the original buffer size, so we could send a dummy buffer of this size to other clients
        # This is important since all XGB clients already prepared a buffer of this size and expect the data
        # to be this size.
        headers = {Constant.HEADER_KEY_ENCRYPTED_DATA: True, Constant.HEADER_KEY_ORIGINAL_BUF_SIZE: len(buffer)}
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEND_BUF, value=encoded, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_HEADERS, value=headers, private=True, sticky=False)

    def _process_after_broadcast(self, fl_ctx: FLContext):
        reply = fl_ctx.get_prop(Constant.PARAM_KEY_REPLY)
        assert isinstance(reply, Shareable)

        has_encrypted_data = reply.get_header(Constant.HEADER_KEY_ENCRYPTED_DATA)
        if not has_encrypted_data:
            return

        if self.clear_ghs:
            # this is the root rank
            # TBD: assume MPI requires the original buffer to be sent back to it.
            fl_ctx.set_prop(key=Constant.PARAM_KEY_RCV_BUF, value=self.original_gh_buffer, private=True, sticky=False)
            return

        # this is a receiving non-label client
        # the rcv_buf contains encrypted gh values
        encoded_gh_str = fl_ctx.get_prop(Constant.PARAM_KEY_RCV_BUF)
        self.public_key, self.encrypted_ghs = decode_encrypted_data(encoded_gh_str)

        original_buf_size = reply.get_header(Constant.HEADER_KEY_ORIGINAL_BUF_SIZE)

        # send a dummy buffer of original size to the XGB client since it is expecting data to be this size
        dummy_buf = os.urandom(original_buf_size)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_RCV_BUF, value=dummy_buf, private=True, sticky=False)

    def _process_before_all_gather_v(self, fl_ctx: FLContext):
        buffer = fl_ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)
        aggr_ctx = self.data_converter.decode_aggregation_context(buffer)

        if not aggr_ctx:
            # this AllGatherV is irrelevant to secure processing
            return

        if not self.feature_masks:
            # the feature contexts only need to be set once
            if not aggr_ctx.features:
                return self._abort("missing features in aggregation context", fl_ctx)

            # convert to adder-friendly data structure
            feature_masks = []
            for f in aggr_ctx.features:
                feature_masks.append((f.feature_id, f.sample_bins, f.num_bins))
            self.feature_masks = feature_masks

        rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK)
        if not self.encrypted_ghs:
            if not self.clear_ghs:
                # this is non-label client
                return self._abort(f"no encrypted (g, h) values for aggregation in rank {rank}", fl_ctx)
            else:
                # label client - send a dummy of 4 bytes
                # note: we assume that XGB code computes aggr results in clear text
                # alternatively, we could compute aggr result based on self.clear_ghs here - TBD
                fl_ctx.set_prop(
                    key=Constant.PARAM_KEY_SEND_BUF,
                    value=os.urandom(Constant.DUMMY_BUFFER_SIZE),
                    private=True,
                    sticky=False,
                )
            return

        # compute aggregation
        if aggr_ctx.sample_groups:
            groups = []
            for i in range(len(aggr_ctx.sample_groups)):
                groups.append((i, aggr_ctx.sample_groups[i]))
        else:
            groups = None

        aggr_result = self.adder.add(self.encrypted_ghs, self.feature_masks, groups, encode_sum=True)
        encoded_str = encode_feature_aggregations(aggr_result)
        headers = {Constant.HEADER_KEY_ENCRYPTED_DATA: True, Constant.HEADER_KEY_ORIGINAL_BUF_SIZE: len(buffer)}
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEND_BUF, value=encoded_str, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_HEADERS, value=headers, private=True, sticky=False)

    def _decrypt_aggr_result(self, encoded):
        if not isinstance(encoded, str):
            # this is dummy result of the label-client
            return encoded

        encoded_str = encoded
        decoded_aggrs = decode_feature_aggregations(self.public_key, encoded_str)
        aggrs_to_decrypt = [decoded_aggrs[i][2] for i in range(len(decoded_aggrs))]
        decrypted_aggrs = self.decrypter.decrypt(aggrs_to_decrypt)  # this is a list of clear-text GH numbers

        aggr_result = []
        for i in range(len(decoded_aggrs)):
            fid, gid, _ = decoded_aggrs[i]

            # split GH to (G_list, H_list)
            G_list = []
            H_list = []
            clear_aggr = decrypted_aggrs[i]
            for j in range(len(clear_aggr)):
                G, H = split(clear_aggr[j])
                G_list.append(G)
                H_list.append(H)

            aggr_result.append((fid, gid, (G_list, H_list)))
        return self.data_converter.encode_aggregation_result(aggr_result)

    def _process_after_all_gather_v(self, fl_ctx: FLContext):
        reply = fl_ctx.get_prop(Constant.PARAM_KEY_REPLY)
        assert isinstance(reply, Shareable)
        encrypted_data = reply.get_header(Constant.HEADER_KEY_ENCRYPTED_DATA)
        if not encrypted_data:
            return

        rank = fl_ctx.get_prop(Constant.PARAM_KEY_RANK)
        rcv_buf = fl_ctx.get_prop(Constant.PARAM_KEY_RCV_BUF)

        # this rcv_buf is a list of replies from each rank
        if not isinstance(rcv_buf, list):
            return self._abort(f"rank {rank}: expect a list of aggr result but got {type(rcv_buf)}", fl_ctx)
        rank_replies = rcv_buf

        if not self.clear_ghs:
            # non-label clients don't care about the results
            dummy = os.urandom(Constant.DUMMY_BUFFER_SIZE)
            for i in range(len(rank_replies)):
                rank_replies[i] = dummy
        else:
            # rank_replies contain encrypted aggr result!
            for i in range(len(rank_replies)):
                rank_replies[i] = self._decrypt_aggr_result(rank_replies[i])

        result = self.data_converter.encode_all_gather_v(rank_replies)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEND_BUF, value=result, private=True, sticky=False)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.public_key, self.private_key = generate_keys(self.key_length)
            self.encryptor = Encryptor(self.public_key, self.num_workers)
            self.decrypter = Decrypter(self.private_key, self.num_workers)
            self.adder = Adder(self.num_workers)
        elif event_type == Constant.EVENT_BEFORE_BROADCAST:
            self._process_before_broadcast(fl_ctx)
        elif event_type == Constant.EVENT_AFTER_BROADCAST:
            self._process_after_broadcast(fl_ctx)
        elif event_type == Constant.EVENT_BEFORE_ALL_GATHER_V:
            self._process_before_all_gather_v(fl_ctx)
        elif event_type == Constant.EVENT_AFTER_ALL_GATHER_V:
            self._process_after_all_gather_v(fl_ctx)
