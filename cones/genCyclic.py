import os
import argparse
from itertools import permutations

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('n', type=int, help="The number of generators for the cyclic polytope")
    parser.add_argument('d', type=int, help="The dimension of the cyclic polytope")
    parser.add_argument('out', type=str, help="Output filename")

    return parser.parse_args()

def genString(n, d):
    buf = ""
    for i in range(1, n+1):
        for j in range(1, d+1):
            buf += f"{i**j} "
        buf += "1\n"

    return buf

def main():
    # Get user arguments
    args = parseArgs()

    # Error Checking
    if os.path.exists(args.out):
        raise IOError(f"File: {args.out} already exists!")
    assert args.n > 1
    assert args.d > 1

    # Generate file data
    buf = genString(args.n, args.d)

    # Write File
    with open(args.out, "w") as f:
        f.write(buf)

if __name__ == "__main__":
    main()
