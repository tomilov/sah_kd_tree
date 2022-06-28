import argparse
import contextlib
import struct
import sys

PARAMS_FORMAT = '=fffI'
VERTEX_FORMAT = 'fff'
TRIANGLE_FORMAT = VERTEX_FORMAT * 3


@contextlib.contextmanager
def _file_or_stream(filename, /, *, mode):
    if filename is None or filename == '-':
        if 'w' in mode:
            yield sys.stdout
        elif 'r' in mode:
            yield sys.stdin
        else:
            raise RuntimeError("mode should contain either 'r' or 'w'")
    else:
        with open(filename, mode) as f:
            yield f


def main():
    parser = argparse.ArgumentParser(description='Convert fuzzer crash files to obj and output Params to stderr.')
    parser.add_argument('infile', type=lambda infile: _file_or_stream(infile, mode='rb'), help="input binary file or '-' for stdin")
    parser.add_argument('outfile', type=lambda outfile: _file_or_stream(outfile, mode='w'), help="output OBJ file or '-' for stdout")

    args = parser.parse_args()

    with args.infile as crash_file:
        params = crash_file.read(struct.calcsize(PARAMS_FORMAT))
        tris = crash_file.read()

    assert len(tris) > 0
    assert len(tris) % struct.calcsize(TRIANGLE_FORMAT) == 0

    emptiness_factor, traversal_cost, intersection_cost, max_depth = struct.unpack(PARAMS_FORMAT, params)

    print(f'{float.hex(emptiness_factor)}, {float.hex(traversal_cost)}, {float.hex(intersection_cost)}, {max_depth}', file=sys.stderr)
    print(f'{emptiness_factor}, {traversal_cost}, {intersection_cost}, {max_depth}', file=sys.stderr)

    with args.outfile as obj:
        for ax, ay, az, bx, by, bz, cx, cy, cz in struct.iter_unpack(TRIANGLE_FORMAT, tris):
            print(f'v {ax} {ay} {az}', file=obj)
            print(f'v {bx} {by} {bz}', file=obj)
            print(f'v {cx} {cy} {cz}', file=obj)
        print(file=obj)
        for i in range(0, len(tris) // struct.calcsize(TRIANGLE_FORMAT)):
            print(f'f {i * 3 + 1} {i * 3 + 2} {i * 3 + 3}', file=obj)


if __name__ == '__main__':
    main()

