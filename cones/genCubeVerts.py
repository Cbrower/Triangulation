import os
import argparse
from itertools import product

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('N', type=int, help="The dimension of the cone over a cube")
    parser.add_argument('out', type=str, help="Output filename")

    return parser.parse_args()

def genString(N):
    buf = ""
    
    for row in product([0, 1], repeat=N):
        row = list(row)
        row.append(1)
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
