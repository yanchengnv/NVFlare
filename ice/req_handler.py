import fnmatch
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from ice.defs import EventType, PropKey
from nvflare.apis.fl_context import FLContext
from nvflare.widgets.widget import Widget


class RequestHandler(Widget, ABC):

    def __init__(self, supported_topics: Union[str, List[str]]):
        """Constructor

        Args:
            supported_topics: a list of topics or fnmatch patterns
        """
        Widget.__init__(self)
        self.register_event_handler(EventType.REQUEST_RECEIVED, self._handle_req_received_event)
        if isinstance(supported_topics, str):
            supported_topics = [supported_topics]
        elif not isinstance(supported_topics, list):
            raise ValueError(f"supported_topics should be str or List[str] but got {type(supported_topics)}")
        self.supported_topics = supported_topics

    @abstractmethod
    def handle_request(self, topic: str, data: dict, fl_ctx: FLContext) -> dict:
        pass

    def _handle_req_received_event(self, event_type: str, fl_ctx: FLContext):
        self.log_debug(fl_ctx, f"received event {event_type}")
        topic = fl_ctx.get_prop(PropKey.TOPIC)

        # see whether this handler matches the topic
        matched = False
        for p in self.supported_topics:
            if fnmatch.fnmatch(topic, p):
                matched = True

        if not matched:
            return

        data = fl_ctx.get_prop(PropKey.DATA)
        result = self.handle_request(topic, data, fl_ctx)
        if result:
            fl_ctx.set_prop(PropKey.RESULT, result, private=True, sticky=False)
