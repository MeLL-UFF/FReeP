#!/usr/local/bin/python3.5
"""reducer.py"""

import sys


def main():
    count = 0
    # input comes from STDIN
    for line in sys.stdin:
        if line.strip():
            count += 1
            print(line)
    print(count)


if __name__ == "__main__":
    main()
