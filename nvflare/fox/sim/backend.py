# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from nvflare.apis.fl_exception import RunAborted
from nvflare.fox import fox
from nvflare.fox.api.app import App
from nvflare.fox.api.backend import Backend
from nvflare.fox.api.call_opt import CallOption
from nvflare.fox.api.constants import CollabMethodArgName, ContextKey
from nvflare.fox.api.dec import adjust_kwargs
from nvflare.fox.api.gcc import GroupCallContext
from nvflare.fox.api.utils import check_call_args


class _Waiter(threading.Event):

    def __init__(self):
        super().__init__()
        self.result = None


class SimBackend(Backend):

    def __init__(self, target_obj_name: str, target_app: App, target_obj, abort_signal, thread_executor, max_workers: int = 100):
        Backend.__init__(self, abort_signal)
        self.target_obj_name = target_obj_name
        self.target_app = target_app
        self.target_obj = target_obj
        self.executor = thread_executor
        # Separate executor for nested calls to avoid deadlock when call_target is called
        # from within an executor thread. Size it to handle all potential concurrent nested calls.
        # Use max_workers to ensure we don't bottleneck.
        self.nested_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="nested-call")

    def _get_func(self, func_name):
        return self.target_app.find_collab_method(self.target_obj, func_name)

    def call_target(self, target_name: str, call_opt: CallOption, func_name: str, *args, **kwargs):
        func = self._get_func(func_name)
        if not func:
            raise AttributeError(f"{target_name} does not have method '{func_name}' or it is not collab")

        if not callable(func):
            raise AttributeError(f"the method '{func_name}' of {target_name} is not callable")

        expect_result = call_opt.expect_result
        timeout = call_opt.timeout

        if not expect_result:
            # Fire and forget - submit to nested executor and return immediately
            self.nested_executor.submit(self._execute_func, target_name, func_name, func, args, kwargs)
            return None

        # Execute with timeout support using separate executor to avoid deadlock
        future = self.nested_executor.submit(self._execute_func, target_name, func_name, func, args, kwargs)
        
        start_time = time.time()
        while True:
            # Check abort signal
            if self.abort_signal.triggered:
                return RunAborted("job is aborted")
            
            try:
                # Wait for result with short polling interval to check abort signal
                result = future.result(timeout=0.1)
                return result
            except FutureTimeoutError:
                # Check if overall timeout exceeded
                waited = time.time() - start_time
                if waited > timeout:
                    return TimeoutError(f"function {func_name} timed out after {waited} seconds")
                # Otherwise continue polling
            except Exception as ex:
                # Function raised an exception
                return ex

    def _execute_func(self, target_name, func_name, func, args, kwargs):
        """Execute function with preprocessing and result filtering."""
        try:
            ctx, kwargs = self._preprocess(target_name, func_name, func, kwargs)
            result = func(*args, **kwargs)
            result = self.target_app.apply_outgoing_result_filters(target_name, func_name, result, ctx)
            return result
        except Exception as ex:
            raise

    def _preprocess(self, target_name, func_name, func, kwargs):
        caller_ctx = kwargs.pop(CollabMethodArgName.CONTEXT)
        my_ctx = self.target_app.new_context(caller_ctx.caller, caller_ctx.callee)
        kwargs = self.target_app.apply_incoming_call_filters(target_name, func_name, kwargs, my_ctx)

        # make sure the final kwargs conforms to func interface
        obj_itf = self.target_app.get_target_object_collab_interface(self.target_obj_name)
        if not obj_itf:
            raise RuntimeError(f"cannot find collab interface for object {self.target_obj_name}")

        func_itf = obj_itf.get(func_name)
        if not func_itf:
            raise RuntimeError(f"cannot find interface for func '{func_name}' of object {self.target_obj_name}")

        check_call_args(func_name, func_itf, [], kwargs)
        kwargs[CollabMethodArgName.CONTEXT] = my_ctx
        adjust_kwargs(func, kwargs)
        return my_ctx, kwargs

    def _run_func(self, waiter: _Waiter, target_name, func_name, func, args, kwargs):
        try:
            ctx, kwargs = self._preprocess(target_name, func_name, func, kwargs)
            result = func(*args, **kwargs)
            # set_call_context(ctx)

            # apply result filter
            result = self.target_app.apply_outgoing_result_filters(target_name, func_name, result, ctx)
            if waiter:
                waiter.result = result
        except Exception as ex:
            if waiter:
                waiter.result = ex
        finally:
            if waiter:
                waiter.set()

    def call_target_in_group(self, gcc: GroupCallContext, func_name: str, *args, **kwargs):
        target_name = gcc.target_name
        func = self._get_func(func_name)
        if not func:
            raise AttributeError(f"{target_name} does not have method '{func_name}' or it is not collab")

        if not callable(func):
            raise AttributeError(f"the method '{func_name}' of {target_name} is not callable")

        self.executor.submit(self._run_func_in_group, gcc, func_name, args, kwargs)

    def _run_func_in_group(self, gcc: GroupCallContext, func_name, args, kwargs):
        try:
            target_name = gcc.target_name
            result = self.call_target(target_name, gcc.call_opt, func_name, *args, **kwargs)
            gcc.set_result(result)
        except Exception as ex:
            gcc.set_exception(ex)
