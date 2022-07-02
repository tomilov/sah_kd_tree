import argparse
import contextlib
import pywavefront
import struct
import sys

PARAMS_FORMAT = '=fffI'
TRIANGLE_FORMAT = '=fffffffff'


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


def _fuzz2obj(args):
    with args.infile as crash_file:
        params = crash_file.read(struct.calcsize(PARAMS_FORMAT))
        tris = crash_file.read()

    assert len(tris) > 0
    assert len(tris) % struct.calcsize(TRIANGLE_FORMAT) == 0

    emptiness_factor, traversal_cost, intersection_cost, max_depth = struct.unpack(PARAMS_FORMAT, params)

    print(f'{float.hex(emptiness_factor)}, {float.hex(traversal_cost)}, {float.hex(intersection_cost)}, {max_depth}', file=sys.stderr)
    print(f'{emptiness_factor}, {traversal_cost}, {intersection_cost}, {max_depth}', file=sys.stderr)

    with args.outfile as obj_file:
        for ax, ay, az, bx, by, bz, cx, cy, cz in struct.iter_unpack(TRIANGLE_FORMAT, tris):
            print(f'v {ax} {ay} {az}', file=obj_file)
            print(f'v {bx} {by} {bz}', file=obj_file)
            print(f'v {cx} {cy} {cz}', file=obj_file)
        print(file=obj_file)
        for i in range(0, len(tris) // struct.calcsize(TRIANGLE_FORMAT)):
            print(f'f {i * 3 + 1} {i * 3 + 2} {i * 3 + 3}', file=obj_file)


def _parse_params(params):
    class Params:
        def __init__(self, params):
            emptiness_factor, traversal_cost, intersection_cost, max_depth = params.split(',')

            self.emptiness_factor = Params._guess_float(emptiness_factor)
            self.traversal_cost = Params._guess_float(traversal_cost)
            self.intersection_cost = Params._guess_float(intersection_cost)
            self.max_depth = int(max_depth, 0)
            assert self.max_depth == 0xFFFFFFFF

        @staticmethod
        def _guess_float(f):
            try:
                return float(f)
            except ValueError:
                return float.fromhex(f)

    return Params(params)


def _obj2fuzz(args):
    with args.outfile as crash_file:
        params = struct.pack(PARAMS_FORMAT, args.params.emptiness_factor, args.params.traversal_cost, args.params.intersection_cost, args.params.max_depth)
        crash_file.write(params)

        triangles = []
        scene = pywavefront.Wavefront(args.infile, create_materials=True, collect_faces=True)
        for mesh in scene.mesh_list:
            for indices in mesh.faces:
                assert len(indices) == 3
                ax, ay, az = scene.vertices[indices[0]]
                bx, by, bz = scene.vertices[indices[1]]
                cx, cy, cz = scene.vertices[indices[2]]
                triangles.append((ax, ay, az, bx, by, bz, cx, cy, cz))
        triangles.sort()
        for triangle in triangles:
            face = struct.pack(TRIANGLE_FORMAT, *triangle)
            crash_file.write(face)


def main():
    parser = argparse.ArgumentParser(description='Convert fuzzer input files to OBJ.')
    subparsers = parser.add_subparsers(title='Converters.', description='Converters between fuzzer input and OBJ files')

    fuzz2obj = subparsers.add_parser('fuzz2obj')
    fuzz2obj.add_argument('infile', type=lambda infile: _file_or_stream(infile, mode='rb'), help="input fuzzer input file or '-' for stdin")
    fuzz2obj.add_argument('outfile', type=lambda outfile: _file_or_stream(outfile, mode='wt'), help="output OBJ file or '-' for stdout")
    fuzz2obj.set_defaults(handler=_fuzz2obj)

    obj2fuzz = subparsers.add_parser('obj2fuzz')
    obj2fuzz.add_argument('--params', type=_parse_params, help="Params in form 'emptinessFactor,traversalCost,intersectionCos,maxDepth', maxDepth currently should be 0xFFFFFFFF or 4294967295")
    obj2fuzz.add_argument('infile', help="input OBJ file file or '-' for stdin")
    obj2fuzz.add_argument('outfile', type=lambda outfile: _file_or_stream(outfile, mode='wb'), help="output fuzzer input file or '-' for stdout")
    obj2fuzz.set_defaults(handler=_obj2fuzz)

    args = parser.parse_args()
    args.handler(args)


if __name__ == '__main__':
    main()

