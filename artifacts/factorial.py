#!/usr/bin/env python3
"""
Factorial with optimal time and space complexity.

- Uses math.factorial (CPython C-optimized) for best performance.
- Provides a pure-Python iterative fallback for reference.
"""

from __future__ import annotations
import math


def factorial(n: int) -> int:
    """
    Return n! for a non-negative integer n.

    Time: O(n) multiplications in the theoretical model (best available).
    Space: O(1) auxiliary space (ignoring integer growth).
    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be >= 0")
    # math.factorial is highly optimized in C
    return math.factorial(n)


def factorial_iter(n: int) -> int:
    """
    Pure-Python iterative factorial, avoids recursion to keep O(1) stack.
    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be >= 0")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python factorial.py <non-negative-integer>")
        sys.exit(1)
    try:
        n = int(sys.argv[1])
    except ValueError:
        print("Error: input must be an integer.")
        sys.exit(1)
    print(factorial(n))
