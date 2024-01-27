import time

from configuration.my_logger import logger


def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        full_time = te - ts
        print(f'\n{f.__name__} took: {full_time} sec')
        return result

    return timed