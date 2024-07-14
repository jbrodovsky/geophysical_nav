"""
Test module for getting used to mypyc
"""

import time
import sys


def fib(n: int) -> int:
    if n <= 1:
        return n
    else:
        return fib(n - 2) + fib(n - 1)


def main(n: int) -> None:
    t0: float = time.time()
    fib(n)
    print(time.time() - t0)


if __name__ == "__main__":
    # take a command line input arguement for n that will be passed to main() and fib(n)
    main(int(sys.argv[1]))

