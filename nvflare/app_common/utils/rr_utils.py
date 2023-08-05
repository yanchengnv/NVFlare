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

from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.app_common.abstract.learnable import Learnable


class RROrder:

    FIXED = "fixed"
    RANDOM = "random"


class RRConstant:

    TASK_NAME_RR = "rr"
    TASK_NAME_SUBMIT_RESULT = "submit_result"
    ORDER = "order"
    CLIENTS = "clients"
    NUM_ROUNDS = "num_rounds"
    TASK_DATA = "task_data"
    LAST_ROUND = "last_round"
    ROUND_START_TIME = "round_start_time"
    ROUND_END_TIME = "round_end_time"
    FINAL_RESULT = "final_result"
    REASON = "reason"
    TOPIC_REPORT_STATUS = "rr.report_status"
    TOPIC_FAILURE = "rr.failure"
    TOPIC_LEARN = "rr.learn"


class StatusReport:
    def __init__(self, last_round=0, start_time=None, end_time=None, final_result=None):
        self.last_round = last_round
        self.start_time = start_time
        self.end_time = end_time
        self.final_result = final_result

    def to_shareable(self) -> Shareable:
        result = Shareable()
        result[RRConstant.LAST_ROUND] = self.last_round
        if self.start_time:
            result[RRConstant.ROUND_START_TIME] = self.start_time
        if self.end_time:
            result[RRConstant.ROUND_END_TIME] = self.end_time
        if self.final_result:
            result[RRConstant.FINAL_RESULT] = self.final_result
        return result


def status_report_from_shareable(d: Shareable) -> StatusReport:
    last_round = d.get(RRConstant.LAST_ROUND)
    if last_round is None:
        raise ValueError(f"missing {RRConstant.LAST_ROUND}")
    start_time = d.get(RRConstant.ROUND_START_TIME)
    end_time = d.get(RRConstant.ROUND_END_TIME)
    final_result = d.get(RRConstant.FINAL_RESULT)

    return StatusReport(
        last_round=last_round,
        start_time=start_time,
        end_time=end_time,
        final_result=final_result,
    )


def execution_failure(reason: str) -> Shareable:
    s = Shareable()
    s[RRConstant.REASON] = reason
    return s


def learnable_to_shareable(learnable: Learnable) -> Shareable:
    s = Shareable()
    s.update(learnable)
    return s


def shareable_to_learnable(shareable: Shareable) -> Learnable:
    learnable = Learnable()
    learnable.update(shareable)
    learnable.pop(ReservedHeaderKey.HEADERS, None)
    return learnable
