import time

# times time to a function call
def taketime(text=None):
    def wrap(f):
        def invoke_func(*args, **kwargs):
            start_time = time.time()
            ret = f(*args, **kwargs)
            elapsed_time = time.time() - start_time
            result_str = '#results %d; ' % len(ret) if hasattr(ret, '__len__') else ''
            print("done %s (%stook %.2fs)" % (text, result_str, elapsed_time))
            return ret
        return invoke_func
    return wrap
