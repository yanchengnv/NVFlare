from abc import abstractmethod

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.tie.connector import Connector
from nvflare.app_common.xgb.defs import Constant
from nvflare.fuel.utils.validation_utils import check_non_negative_int, check_positive_int


class XGBServerConnector(Connector):
    """
    XGBServerAdaptor specifies commonly required methods for server adaptor implementations.
    """

    def __init__(self, in_process):
        Connector.__init__(self, in_process)
        self.world_size = None
        # set up operation handlers
        self.op_table = {
            Constant.OP_ALL_GATHER: self._process_all_gather,
            Constant.OP_ALL_GATHER_V: self._process_all_gather_v,
            Constant.OP_ALL_REDUCE: self._process_all_reduce,
            Constant.OP_BROADCAST: self._process_broadcast,
        }

    def configure(self, config: dict, fl_ctx: FLContext):
        """Called by XGB Controller to configure the target.

        The world_size is a required config parameter.

        Args:
            config: config data
            fl_ctx: FL context

        Returns: None

        """
        ws = config.get(Constant.CONF_KEY_WORLD_SIZE)
        if not ws:
            raise RuntimeError("world_size is not configured")

        check_positive_int(Constant.CONF_KEY_WORLD_SIZE, ws)
        self.world_size = ws

    def _process_all_gather(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """This is the op handler for Allgather.

        Args:
            request: the request containing op params
            fl_ctx: FL context

        Returns: a Shareable containing operation result

        """
        rank = request.get(Constant.PARAM_KEY_RANK)
        seq = request.get(Constant.PARAM_KEY_SEQ)
        send_buf = request.get(Constant.PARAM_KEY_SEND_BUF)
        rcv_buf = self.all_gather(rank, seq, send_buf, fl_ctx)
        reply = Shareable()
        reply[Constant.PARAM_KEY_RCV_BUF] = rcv_buf
        return reply

    def _process_all_gather_v(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """This is the op handler for AllgatherV.

        Args:
            request: the request containing op params
            fl_ctx: FL context

        Returns: a Shareable containing operation result

        """
        rank = request.get(Constant.PARAM_KEY_RANK)
        seq = request.get(Constant.PARAM_KEY_SEQ)
        send_buf = request.get(Constant.PARAM_KEY_SEND_BUF)

        fl_ctx.set_prop(key=Constant.PARAM_KEY_RANK, value=rank, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEQ, value=seq, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEND_BUF, value=send_buf, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_REQUEST, value=request, private=True, sticky=False)
        self.fire_event(Constant.EVENT_BEFORE_ALL_GATHER_V, fl_ctx)

        send_buf = fl_ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)

        rcv_buf = self.all_gather_v(rank, seq, send_buf, fl_ctx)
        reply = Shareable()

        fl_ctx.set_prop(key=Constant.PARAM_KEY_REPLY, value=reply, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_RCV_BUF, value=rcv_buf, private=True, sticky=False)
        self.fire_event(Constant.EVENT_AFTER_ALL_GATHER_V, fl_ctx)
        rcv_buf = fl_ctx.get_prop(Constant.PARAM_KEY_RCV_BUF)

        reply[Constant.PARAM_KEY_RCV_BUF] = rcv_buf
        return reply

    def _process_all_reduce(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """This is the op handler for Allreduce.

        Args:
            request: the request containing op params
            fl_ctx: FL context

        Returns: a Shareable containing operation result

        """
        rank = request.get(Constant.PARAM_KEY_RANK)
        seq = request.get(Constant.PARAM_KEY_SEQ)
        send_buf = request.get(Constant.PARAM_KEY_SEND_BUF)
        data_type = request.get(Constant.PARAM_KEY_DATA_TYPE)
        reduce_op = request.get(Constant.PARAM_KEY_REDUCE_OP)
        rcv_buf = self.all_reduce(rank, seq, data_type, reduce_op, send_buf, fl_ctx)
        reply = Shareable()
        reply[Constant.PARAM_KEY_RCV_BUF] = rcv_buf
        return reply

    def _process_broadcast(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """This is the op handler for Broadcast.

        Args:
            request: the request containing op params
            fl_ctx: FL context

        Returns: a Shareable containing operation result

        """
        rank = request.get(Constant.PARAM_KEY_RANK)
        seq = request.get(Constant.PARAM_KEY_SEQ)
        send_buf = request.get(Constant.PARAM_KEY_SEND_BUF)
        root = request.get(Constant.PARAM_KEY_ROOT)

        fl_ctx.set_prop(key=Constant.PARAM_KEY_RANK, value=rank, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEQ, value=seq, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_ROOT, value=root, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEND_BUF, value=send_buf, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_REQUEST, value=request, private=True, sticky=False)
        self.fire_event(Constant.EVENT_BEFORE_BROADCAST, fl_ctx)

        send_buf = fl_ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)
        rcv_buf = self.broadcast(rank, seq, root, send_buf, fl_ctx)
        reply = Shareable()

        fl_ctx.set_prop(key=Constant.PARAM_KEY_REPLY, value=reply, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_RCV_BUF, value=rcv_buf, private=True, sticky=False)
        self.fire_event(Constant.EVENT_AFTER_BROADCAST, fl_ctx)
        rcv_buf = fl_ctx.get_prop(Constant.PARAM_KEY_RCV_BUF)
        reply[Constant.PARAM_KEY_RCV_BUF] = rcv_buf
        return reply

    @abstractmethod
    def all_gather(self, rank: int, seq: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        """Called by the XGB Controller to perform Allgather operation, per XGBoost spec.

        Args:
            rank: rank of the calling client
            seq: sequence number of the request
            send_buf: operation input data
            fl_ctx: FL context

        Returns: operation result

        """
        pass

    @abstractmethod
    def all_gather_v(self, rank: int, seq: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        """Called by the XGB Controller to perform AllgatherV operation, per XGBoost spec.

        Args:
            rank: rank of the calling client
            seq: sequence number of the request
            send_buf: input data
            fl_ctx: FL context

        Returns: operation result

        """
        pass

    @abstractmethod
    def all_reduce(
        self,
        rank: int,
        seq: int,
        data_type: int,
        reduce_op: int,
        send_buf: bytes,
        fl_ctx: FLContext,
    ) -> bytes:
        """Called by the XGB Controller to perform Allreduce operation, per XGBoost spec.

        Args:
            rank: rank of the calling client
            seq: sequence number of the request
            data_type: data type of the input
            reduce_op: reduce operation to be performed
            send_buf: input data
            fl_ctx: FL context

        Returns: operation result

        """
        pass

    @abstractmethod
    def broadcast(self, rank: int, seq: int, root: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        """Called by the XGB Controller to perform Broadcast operation, per XGBoost spec.

        Args:
            rank: rank of the calling client
            seq: sequence number of the request
            root: root rank of the broadcast
            send_buf: input data
            fl_ctx: FL context

        Returns: operation result

        """
        pass

    def process_app_request(self, op: str, request: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        stopped, ec = self._is_stopped()
        if stopped:
            raise RuntimeError(f"dropped XGB request '{op}' since connector is already stopped {ec=}")

        # find and call the op handlers
        process_f = self.op_table.get(op)
        if process_f is None:
            raise RuntimeError(f"invalid op '{op}' from XGB request")

        if not callable(process_f):
            # impossible but we must declare process_f to be callable; otherwise PyCharm will complain about
            # process_f(request, fl_ctx).
            raise RuntimeError(f"op handler for {op} is not callable")

        reply = process_f(request, fl_ctx)
        self.log_info(fl_ctx, f"received reply for '{op}'")
        reply.set_header(Constant.MSG_KEY_XGB_OP, op)
        return reply


class XGBClientConnector(Connector):
    """
    XGBClientConnector specifies commonly required methods for client connector implementations.
    """

    def __init__(self, in_process, per_msg_timeout, tx_timeout):
        """Constructor of XGBClientAdaptor"""
        Connector.__init__(self, in_process)
        self.per_msg_timeout = per_msg_timeout
        self.tx_timeout = tx_timeout
        self.stopped = False
        self.rank = None
        self.num_rounds = None
        self.world_size = None

    def configure(self, config: dict, fl_ctx: FLContext):
        """Called by XGB Executor to configure the target.

        The rank, world size, and number of rounds are required config parameters.

        Args:
            config: config data
            fl_ctx: FL context

        Returns: None

        """
        ws = config.get(Constant.CONF_KEY_WORLD_SIZE)
        if not ws:
            raise RuntimeError("world_size is not configured")

        check_positive_int(Constant.CONF_KEY_WORLD_SIZE, ws)
        self.world_size = ws

        rank = config.get(Constant.CONF_KEY_RANK)
        if rank is None:
            raise RuntimeError("rank is not configured")

        check_non_negative_int(Constant.CONF_KEY_RANK, rank)
        self.rank = rank

        num_rounds = config.get(Constant.CONF_KEY_NUM_ROUNDS)
        if num_rounds is None:
            raise RuntimeError("num_rounds is not configured")

        check_positive_int(Constant.CONF_KEY_NUM_ROUNDS, num_rounds)
        self.num_rounds = num_rounds

    def _send_request(self, op: str, req: Shareable) -> (bytes, Shareable):
        """Send XGB operation request to the FL server via FLARE message.

        Args:
            op: the XGB operation
            req: operation data

        Returns: operation result

        """
        reply = self.send_request(
            op=op,
            target=None,  # server
            request=req,
            per_msg_timeout=self.per_msg_timeout,
            tx_timeout=self.tx_timeout,
            fl_ctx=None,
        )
        if not isinstance(reply, Shareable):
            raise RuntimeError(f"invalid reply for op {op}: expect Shareable but got {type(reply)}")

        rc = reply.get_return_code()
        if rc != ReturnCode.OK:
            raise RuntimeError(f"error reply for op {op}: {rc=}")
        rcv_buf = reply.get(Constant.PARAM_KEY_RCV_BUF)
        return rcv_buf, reply

    def _send_all_gather(self, rank: int, seq: int, send_buf: bytes) -> (bytes, Shareable):
        """This method is called by a concrete client adaptor to send Allgather operation to the server.

        Args:
            rank: rank of the client
            seq: sequence number of the request
            send_buf: input data

        Returns: operation result

        """
        req = Shareable()
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_ALL_GATHER, req)

    def _send_all_gather_v(self, rank: int, seq: int, send_buf: bytes, headers=None) -> (bytes, Shareable):
        req = Shareable()
        self._add_headers(req, headers)
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_ALL_GATHER_V, req)

    def _do_all_gather_v(self, rank: int, seq: int, send_buf: bytes) -> (bytes, Shareable):
        """This method is called by a concrete client adaptor to send AllgatherV operation to the server.

        Args:
            rank: rank of the client
            seq: sequence number of the request
            send_buf: operation input

        Returns: operation result

        """
        fl_ctx = self.engine.new_context()
        fl_ctx.set_prop(key=Constant.PARAM_KEY_RANK, value=rank, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEQ, value=seq, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEND_BUF, value=send_buf, private=True, sticky=False)
        self.fire_event(Constant.EVENT_BEFORE_ALL_GATHER_V, fl_ctx)

        send_buf = fl_ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)
        rcv_buf, reply = self._send_all_gather_v(
            rank=rank,
            seq=seq,
            send_buf=send_buf,
            headers=fl_ctx.get_prop(Constant.PARAM_KEY_HEADERS),
        )

        fl_ctx.set_prop(key=Constant.PARAM_KEY_RCV_BUF, value=rcv_buf, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_REPLY, value=reply, private=True, sticky=False)
        self.fire_event(Constant.EVENT_AFTER_ALL_GATHER_V, fl_ctx)
        return fl_ctx.get_prop(Constant.PARAM_KEY_RCV_BUF)

    def _send_all_reduce(
        self, rank: int, seq: int, data_type: int, reduce_op: int, send_buf: bytes
    ) -> (bytes, Shareable):
        """This method is called by a concrete client adaptor to send Allreduce operation to the server.

        Args:
            rank: rank of the client
            seq: sequence number of the request
            data_type: data type of the input
            reduce_op: reduce operation to be performed
            send_buf: operation input

        Returns: operation result

        """
        req = Shareable()
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_DATA_TYPE] = data_type
        req[Constant.PARAM_KEY_REDUCE_OP] = reduce_op
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_ALL_REDUCE, req)

    def _send_broadcast(self, rank: int, seq: int, root: int, send_buf: bytes, headers=None) -> (bytes, Shareable):
        req = Shareable()
        self._add_headers(req, headers)
        req[Constant.PARAM_KEY_RANK] = rank
        req[Constant.PARAM_KEY_SEQ] = seq
        req[Constant.PARAM_KEY_ROOT] = root
        req[Constant.PARAM_KEY_SEND_BUF] = send_buf
        return self._send_request(Constant.OP_BROADCAST, req)

    def _do_broadcast(self, rank: int, seq: int, root: int, send_buf: bytes) -> bytes:
        """This method is called by a concrete client adaptor to send Broadcast operation to the server.

        Args:
            rank: rank of the client
            seq: sequence number of the request
            root: root rank of the broadcast
            send_buf: operation input

        Returns: operation result

        """
        fl_ctx = self.engine.new_context()
        fl_ctx.set_prop(key=Constant.PARAM_KEY_RANK, value=rank, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEQ, value=seq, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_ROOT, value=root, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_SEND_BUF, value=send_buf, private=True, sticky=False)
        self.fire_event(Constant.EVENT_BEFORE_BROADCAST, fl_ctx)

        send_buf = fl_ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)
        rcv_buf, reply = self._send_broadcast(
            rank=rank,
            seq=seq,
            root=root,
            send_buf=send_buf,
            headers=fl_ctx.get_prop(Constant.PARAM_KEY_HEADERS),
        )

        fl_ctx.set_prop(key=Constant.PARAM_KEY_RCV_BUF, value=rcv_buf, private=True, sticky=False)
        fl_ctx.set_prop(key=Constant.PARAM_KEY_REPLY, value=reply, private=True, sticky=False)
        self.fire_event(Constant.EVENT_AFTER_BROADCAST, fl_ctx)
        return fl_ctx.get_prop(Constant.PARAM_KEY_RCV_BUF)

    @staticmethod
    def _add_headers(req: Shareable, headers: dict):
        if not headers:
            return

        for k, v in headers.items():
            req.set_header(k, v)
