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


class Constant:

    TASK_NAME_CONFIGURE = "configure"
    TASK_NAME_START = "start"
    TASK_NAME_SUBMIT_RESULT = "submit_result"

    ORDER = "cw.order"
    CLIENTS = "cw.clients"
    CLIENT_ORDER = "cw.client_order"
    LAST_ROUND = "cw.last_round"
    START_TIME = "cw.start_time"
    END_TIME = "cw.end_time"
    LAST_RESULT = "cw.last_result"
    RESULT_TYPE = "cw.result_type"
    ALL_DONE = "cw.all_done"
    REASON = "cw.reason"
    AGGR_CLIENTS = "cw.aggr_clients"
    TRAIN_CLIENTS = "cw.train_clients"
    AGGREGATOR = "cw.aggr"
    BEST_METRIC = "cw.best_metric"
    CONFIG = "cw.config"

    TOPIC_REPORT_STATUS = "cw.report_status"
    TOPIC_FAILURE = "cw.failure"
    TOPIC_LEARN = "cw.learn"
    TOPIC_RESULT = "cw.result"


class RROrder:

    FIXED = "fixed"
    RANDOM = "random"


class StatusReport:
    def __init__(self, last_round=0, start_time=None, end_time=None, best_metric=None, all_done=False):
        self.last_round = last_round
        self.start_time = start_time
        self.end_time = end_time
        self.best_metric = best_metric
        self.all_done = all_done

    def to_shareable(self) -> Shareable:
        result = Shareable()
        result[Constant.LAST_ROUND] = self.last_round
        result[Constant.ALL_DONE] = self.all_done

        if self.start_time:
            result[Constant.START_TIME] = self.start_time
        if self.end_time:
            result[Constant.END_TIME] = self.end_time
        if self.best_metric:
            result[Constant.BEST_METRIC] = self.best_metric
        return result

    def __eq__(self, other):
        if not isinstance(other, StatusReport):
            # don't attempt to compare against unrelated types
            return ValueError(f"cannot compare to object of type {type(other)}")

        return (
            self.last_round == other.last_round
            and self.start_time == other.start_time
            and self.end_time == other.end_time
            and self.all_done == other.all_done
            and self.best_metric == other.best_metric
        )


def status_report_from_shareable(d: Shareable) -> StatusReport:
    last_round = d.get(Constant.LAST_ROUND)
    if last_round is None:
        raise ValueError(f"missing {Constant.LAST_ROUND} in status report")
    start_time = d.get(Constant.START_TIME)
    end_time = d.get(Constant.END_TIME)
    all_done = d.get(Constant.ALL_DONE)
    best_metric = d.get(Constant.BEST_METRIC)

    return StatusReport(
        last_round=last_round, start_time=start_time, end_time=end_time, best_metric=best_metric, all_done=all_done
    )


def execution_failure(reason: str) -> Shareable:
    s = Shareable()
    s[Constant.REASON] = reason
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


def rotate_to_front(item, items: list):
    num_items = len(items)
    idx = items.index(item)
    if idx != 0:
        new_list = [None] * num_items
        for i in range(num_items):
            new_pos = i - idx
            if new_pos < 0:
                new_pos += num_items
            new_list[new_pos] = items[i]

        for i in range(num_items):
            items[i] = new_list[i]
