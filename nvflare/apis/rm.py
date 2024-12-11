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
from abc import ABC, abstractmethod

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable

PROP_KEY_DEBUG_INFO = "RM.DEBUG_INFO"


class RMEngine(ABC):
    @abstractmethod
    def send_reliable_request(
        self,
        target: str,
        channel: str,
        topic: str,
        request: Shareable,
        per_msg_timeout: float,
        tx_timeout: float,
        fl_ctx: FLContext,
        optional=False,
        secure=False,
    ) -> Shareable:
        """Send a reliable request.

        Args:
            target: the destination of the request
            channel: the message channel
            topic: the message topic
            request: the request to be sent
            per_msg_timeout: Number of seconds to wait for each message before timing out.
            tx_timeout: Timeout for the entire transaction
            fl_ctx: FLContext object
            secure: whether to use P2P security for this message
            optional: whether the message is optional

        Returns: reply from the target

        """
        pass

    @abstractmethod
    def register_reliable_request_handler(self, channel: str, topic: str, handler_f, **handler_kwargs):
        """Register request handler for reliable requests.

        Args:
            channel: message channel
            topic: message topic
            handler_f: function for handling the requests
            **handler_kwargs: kwargs to be passed to the handler function

        Returns: None

        """
        pass

    @abstractmethod
    def shutdown_reliable_messenger(self):
        """Shutdown reliable messenger.

        Returns: None

        """
        pass


def reliable_request_handler_signature(
    channel: str, topic: str, request: Shareable, fl_ctx: FLContext, **handler_kwargs
) -> Shareable:
    """This is the signature of reliable request handler function

    Args:
        channel: message channel
        topic: message topic
        request: the request to be handled
        fl_ctx: FLContext object
        **handler_kwargs: kwargs registered with the handler function

    Returns: a Shareable object

    """
    pass
