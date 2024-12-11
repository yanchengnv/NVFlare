# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import enum
from abc import ABC, abstractmethod
from typing import Dict

from .fl_context import FLContext
from .shareable import Shareable


class MessageSendStatus(enum.Enum):

    OK = "ok"  # message sent and response received
    TIMEOUT = "timeout"  # message sent but no response received
    FAILURE = "failure"  # failed to send message
    REPLY_ERROR = "reply_error"  # error in reply


def aux_request_handle_func_signature(topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
    """This is the signature of the message_handle_func.

    The message_handle_func is a callback function that is registered to handle an aux request of a specific topic.
    Any implementation of a message_handle_func must follow this signature.

    Example from the client runner:
        engine.register_aux_message_handler(topic=ReservedTopic.END_RUN, message_handle_func=self._handle_end_run)

    Args:
        topic: topic of the message to be handled
        request: the message data to be handled
        fl_ctx: FL context

    Returns: a Shareable response to the requester

    """
    pass


class AuxMessenger(ABC):
    @abstractmethod
    def register_aux_message_handler(self, topic: str, message_handle_func):
        """Register aux message handling function with specified topics.

        Exception is raised when:
            a handler is already registered for the topic;
            bad topic - must be a non-empty string
            bad message_handle_func - must be callable

        Implementation Note:
            This method should simply call the ServerAuxRunner's register_aux_message_handler method.

        Args:
            topic: the topic to be handled by the func
            message_handle_func: the func to handle the message. Must follow aux_message_handle_func_signature.

        """
        pass

    @abstractmethod
    def send_aux_request(
        self,
        targets: [],
        topic: str,
        request: Shareable,
        timeout: float,
        fl_ctx: FLContext,
        optional=False,
        secure=False,
    ) -> dict:
        """Send a request to specified clients via the aux channel.

        Implementation: simply calls the AuxRunner's send_aux_request method.

        Args:
            targets: target clients. None or empty list means all clients.
            topic: topic of the request.
            request: request to be sent
            timeout: number of secs to wait for replies. 0 means fire-and-forget.
            fl_ctx: FL context
            optional: whether this message is optional
            secure: send the aux request in a secure way

        Returns: a dict of replies (client name => reply Shareable)

        """
        pass

    @abstractmethod
    def multicast_aux_requests(
        self,
        topic: str,
        target_requests: Dict[str, Shareable],
        timeout: float,
        fl_ctx: FLContext,
        optional: bool = False,
        secure: bool = False,
    ) -> dict:
        """Send requests to specified clients via the aux channel.

        Implementation: simply calls the AuxRunner's multicast_aux_requests method.

        Args:
            topic: topic of the request
            target_requests: requests of the target clients. Different target can have different request.
            timeout: amount of time to wait for responses. 0 means fire and forget.
            fl_ctx: FL context
            optional: whether this request is optional
            secure: whether to send the aux request in P2P secure

        Returns: a dict of replies (client name => reply Shareable)

        """
        pass

    def fire_and_forget_aux_request(
        self, targets: [], topic: str, request: Shareable, fl_ctx: FLContext, optional=False, secure=False
    ) -> dict:
        return self.send_aux_request(targets, topic, request, 0.0, fl_ctx, optional, secure=secure)
