# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=use-dict-literal
# pylint: disable=use-list-literal
# mypy: disallow-untyped-defs

import argparse
from pathlib import Path


def get_nonempty(parts: str) -> list[str]:
    return list(filter(lambda part: part, parts.split(" ")))


def parse_depfile(content: str) -> dict[str, list[str]]:
    lines = content.splitlines()
    buffer = []
    rules = []
    for line in lines:
        line_continue = line.endswith("\\")
        if line_continue:
            line = line[:-1]
        line = line.lstrip()
        if line:
            buffer.append(line)
        if not line_continue and buffer:
            line = " ".join(buffer)
            buffer = []
            parts = line.split(":")
            if len(parts) not in (1, 2):
                raise RuntimeError(f"{parts}")
            rule = {
                "targets": get_nonempty(parts[0]),
            }
            if len(parts) == 2:
                rule["dependencies"] = get_nonempty(parts[1])
            rules.append(rule)
    return rules


def add_prefix(prefix: Path, paths: list[str]) -> str:
    prefixed_paths: list[str] = []
    for path in paths:
        path = Path(path)
        if not path.is_absolute():
            path = prefix / path
        prefixed_paths.append(str(path))
    return " ".join(prefixed_paths)


def main() -> None:
    parser = argparse.ArgumentParser("Fix paths in depfile")
    parser.add_argument(
        "prefix",
        type=Path,
        help="Filesystem prefix to prepend to targets and dependencies",
    )
    parser.add_argument("depfile_in", type=Path, help="Input depfile")
    parser.add_argument(
        "depfile_out",
        type=Path,
        nargs="?",
        help="Output depfile",
    )
    args = parser.parse_args()

    rules = parse_depfile(args.depfile_in.read_text())
    depfile = args.depfile_out if args.depfile_out else args.depfile_in
    with depfile.open("wt") as file:
        for rule in rules:
            print(add_prefix(args.prefix, rule["targets"]), end="", file=file)
            if "dependencies" in rule:
                print(": ", end="", file=file)
                print(
                    add_prefix(args.prefix, rule["dependencies"]),
                    end="",
                    file=file,
                )
            print(file=file)


if __name__ == "__main__":
    main()
