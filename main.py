import os
from argparse import ArgumentParser

from config import load_config
from treesearch.search import TreeSearch
from utils.log import set_log_level, _ROOT_LOGGER
from utils.path import mkdir

logger = _ROOT_LOGGER.getChild("main")


def main():
    set_log_level(os.getenv("ISGSA_LOG", "INFO"))

    config = load_config()
    out_dir = mkdir(config.out_dir)
    args = get_args()
    if args.init:
        mkdir(out_dir / "workspace")
        return

    user_req_lines: list[str] = []
    print('Enter you request, write "!start" to start:')
    while True:
        line = input("> ")
        if line.lower().strip().startswith("!start"):
            break
        user_req_lines.append(line)

    user_request = "\n".join(user_req_lines)

    logger.info("Starting AutoRecLab...")

    ts = TreeSearch(user_request, config=config)
    ts.run()


def get_args():
    parser = ArgumentParser("AutoRecLab")
    parser.add_argument("--init", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main()
