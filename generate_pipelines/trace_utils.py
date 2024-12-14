import sys
import traceback
from datetime import datetime
from functools import wraps
from io import StringIO
from contextlib import contextmanager


def trace_output(func=None, stdout=True, stderr=True):
    class TracingStream(StringIO):
        def __init__(self, original_stream):
            super().__init__()
            self.original_stream = original_stream

        def write(self, text):
            if text.strip():
                stack = traceback.extract_stack()
                caller_info = '\n'.join((l.strip() for l in traceback.format_list(stack)))
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                output = f"[{timestamp}] {caller_info} | {text}"
                self.original_stream.write(output)

        def flush(self):
            self.original_stream.flush()

    @contextmanager
    def redirect_outputs():
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            if stdout:
                sys.stdout = TracingStream(sys.stdout)
            if stderr:
                sys.stderr = TracingStream(sys.stderr)
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with redirect_outputs():
                return func(*args, **kwargs)
        return wrapper

    # Allow both @trace_output and @trace_output() syntax
    if func is None:
        return decorator
    return decorator(func)
