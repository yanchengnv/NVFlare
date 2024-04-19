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

from nvflare.fuel.f3.drivers.net_utils import MAX_FRAME_SIZE
from nvflare.app_common.tie.defs import Constant as TieConstant


class Constant:

    # task name defaults
    CONFIG_TASK_NAME = TieConstant.CONFIG_TASK_NAME
    START_TASK_NAME = TieConstant.START_TASK_NAME

    # keys of adaptor config parameters
    CONF_KEY_CLIENT_RANKS = "client_ranks"
    CONF_KEY_RANK = "rank"
    CONF_KEY_WORLD_SIZE = "world_size"
    CONF_KEY_NUM_ROUNDS = "num_rounds"

    # default component config values
    CONFIG_TASK_TIMEOUT = TieConstant.CONFIG_TASK_TIMEOUT
    START_TASK_TIMEOUT = TieConstant.START_TASK_TIMEOUT
    XGB_SERVER_READY_TIMEOUT = 10.0

    JOB_STATUS_CHECK_INTERVAL = TieConstant.JOB_STATUS_CHECK_INTERVAL
    MAX_CLIENT_OP_INTERVAL = TieConstant.MAX_CLIENT_OP_INTERVAL
    WORKFLOW_PROGRESS_TIMEOUT = TieConstant.WORKFLOW_PROGRESS_TIMEOUT

    # keys for Shareable between client and server
    MSG_KEY_EXIT_CODE = TieConstant.MSG_KEY_EXIT_CODE
    MSG_KEY_XGB_OP = TieConstant.MSG_KEY_OP

    # XGB operation names
    OP_ALL_GATHER = "all_gather"
    OP_ALL_GATHER_V = "all_gather_v"
    OP_ALL_REDUCE = "all_reduce"
    OP_BROADCAST = "broadcast"

    # XGB operation codes
    OPCODE_NONE = 0
    OPCODE_ALL_GATHER = 1
    OPCODE_ALL_GATHER_V = 2
    OPCODE_ALL_REDUCE = 3
    OPCODE_BROADCAST = 4
    OPCODE_DONE = 99

    # XGB operation error codes
    ERR_OP_MISMATCH = -1
    ERR_INVALID_RANK = -2
    ERR_NO_CLIENT_FOR_RANK = -3
    ERR_TARGET_ERROR = -4

    EXIT_CODE_CANT_START = 101

    # XGB operation parameter keys
    PARAM_KEY_RANK = "xgb.rank"
    PARAM_KEY_SEQ = "xgb.seq"
    PARAM_KEY_SEND_BUF = "xgb.send_buf"
    PARAM_KEY_DATA_TYPE = "xgb.data_type"
    PARAM_KEY_REDUCE_OP = "xgb.reduce_op"
    PARAM_KEY_ROOT = "xgb.root"
    PARAM_KEY_RCV_BUF = "xgb.rcv_buf"
    PARAM_KEY_HEADERS = "xgb.headers"
    PARAM_KEY_REPLY = "xgb.reply"
    PARAM_KEY_REQUEST = "xgb.request"
    PARAM_KEY_EVENT = "xgb.event"

    APP_CTX_SERVER_ADDR = "server_addr"
    APP_CTX_PORT = "port"
    APP_CTX_CLIENT_NAME = "client_name"
    APP_CTX_NUM_ROUNDS = "num_rounds"
    APP_CTX_WORLD_SIZE = "world_size"
    APP_CTX_RANK = "rank"
    APP_CTX_TB_DIR = "tb_dir"
    APP_CTX_MODEL_DIR = "model_dir"

    EVENT_BEFORE_BROADCAST = "xgb.before_broadcast"
    EVENT_AFTER_BROADCAST = "xgb.after_broadcast"
    EVENT_BEFORE_ALL_GATHER_V = "xgb.before_all_gather_v"
    EVENT_AFTER_ALL_GATHER_V = "xgb.after_all_gather_v"
    EVENT_REQ_FAILED = "xgb.req_failed"

    HEADER_KEY_ENCRYPTED_DATA = "xgb.encrypted_data"
    HEADER_KEY_ORIGINAL_BUF_SIZE = "xgb.original_buf_size"

    DUMMY_BUFFER_SIZE = 4


GRPC_DEFAULT_OPTIONS = [
    ("grpc.max_send_message_length", MAX_FRAME_SIZE),
    ("grpc.max_receive_message_length", MAX_FRAME_SIZE),
]
