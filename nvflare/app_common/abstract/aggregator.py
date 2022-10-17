# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class Aggregator(FLComponent, ABC):

    def reset(self, fl_ctx: FLContext):
        """Reset the internal state of the aggregator.

        Args:
            fl_ctx: FLContext

        Returns:

        """
        pass

    @abstractmethod
    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Accept the shareable submitted by the client.

        Args:
            shareable: submitted Shareable object
            fl_ctx: FLContext

        Returns:
            first boolean to indicate if the contribution has been accepted.

        """
        pass

    @abstractmethod
    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Perform the aggregation for all the received Shareable from the clients.

        Args:
            fl_ctx: FLContext

        Returns:
            shareable
        """
        pass
