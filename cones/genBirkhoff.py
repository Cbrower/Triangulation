import os
import argparse
from itertools import permutations

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('N', type=int, help="The dimension of the birkhoff polytope")
    parser.add_argument('out', type=str, help="Output filename")

    return parser.parse_args()

def genString(N):
    perm = permutations([i for i in range(N)])

    buf = ""
    for p in perm:
        row = [0 for i in range(N*N)]
        row.append(1)
        for i in range(len(p)):
            row[i*N + p[i]] = 1
        buf += " ".join(str(e) for e in row) + "\n"

    return buf

def main():
    # Get user arguments
    args = parseArgs()

    # Error Checking
    if os.path.exists(args.out):
        raise IOError(f"File: {args.out} already exists!")
    assert args.N > 1

    # Generate file data
    buf = genString(args.N)

    # Write File
    with open(args.out, "w") as f:
        f.write(buf)

if __name__ == "__main__":
    main()
