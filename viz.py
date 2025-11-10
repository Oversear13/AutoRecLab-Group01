import pickle
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

from anytree.exporter.dotexporter import UniqueDotExporter

from treesearch.node import Node
from utils.path import mkdir


def render_trees(nodes: list[Node], output_dir: Path):
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        tmp_file = tmp_dir / "tmp.dot"

        for idx, dn in enumerate(nodes):
            e = UniqueDotExporter(dn)
            e.to_dotfile(tmp_file)

            for t in ["png", "svg", "pdf"]:
                t_dir = mkdir(output_dir / t)
                out_file = t_dir / f"tree{idx}.{t}"

                subprocess.run(["dot", str(tmp_file), "-T", t, "-o", str(out_file)])


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-pkl", default="./save.pkl")
    parser.add_argument("-o", "--output-dir", default="./tree_render")
    args = parser.parse_args()

    if not Path(args.input_pkl).exists():
        print(f'Input file "{args.input_pkl}" does not exist!')
        return

    output_dir = mkdir(args.output_dir)

    with open(args.input_pkl, "rb") as f:
        nodes: list[Node] = pickle.load(f)
        render_trees(nodes, output_dir)


if __name__ == "__main__":
    main()
