class PropKey:
    TOPIC = "topic"
    DATA = "data"
    STATUS = "status"
    ATTACHMENTS = "attachments"
    FQCN = "fqcn"
    FILE_REF_ID = "file_ref_id"
    FILE_LOCATION = "file_location"
    FILE_ERROR = "file_error"
    RESULT = "result"
    DETAIL = "detail"


class EventType:
    REQUEST_RECEIVED = "ice.request_received"


class StatusCode:
    OK = "ok"
    NO_REPLY = "no_reply"
    ERROR = "error"
    NOT_READY = "not_ready"


REQUEST_TOPIC = "ice.request"
CONFIG_TASK_NAME = "ice_config"
