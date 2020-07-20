"""This script documents useful decorators"""

import logging
import time

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

def clock(func):
    """Decorator for profiling purpose"""
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
        # print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        print('[%0.8fs] %s -> %r' % (elapsed, name, result))
        return result

    return clocked


def dump_func_name(func):
    """This decorator prints out function name when it is called

    Args:
        func:

    Returns:

    """
    def echo_func(*func_args, **func_kwargs):
        logging.debug('### Start func: {}'.format(func.__name__))
        return func(*func_args, **func_kwargs)
    return echo_func


def clock_test():
    @dump_func_name
    @clock
    def fibonacci(n):
        if n < 2:
            return n
        return fibonacci(n - 2) + fibonacci(n - 1)
    print(fibonacci(6))


if __name__ == '__main__':
    clock_test()