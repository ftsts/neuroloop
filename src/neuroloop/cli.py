import argparse
from neuroloop.scripts.ftsts_open_loop import run as run_open_loop
from neuroloop.scripts.ftsts_closed_loop import run as run_closed_loop


def cmd_run(args: argparse.Namespace) -> None:
    "nl run ..."

    assert args.open or args.closed

    if args.open:
        run_open_loop()

    if args.closed:
        run_closed_loop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nl",  # neuroloop
        description="NeuroLoop: RL for DBS control",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # nl run ...
    run_parser = subparsers.add_parser(
        "run",
        help="Run example simulation scripts",
    )
    run_parser.add_argument(
        "-o",
        "--open",
        action="store_true",
        help="Run the open-loop script",
    )
    run_parser.add_argument(
        "-c",
        "--closed",
        action="store_true",
        help="Run the closed-loop script",
    )
    run_parser.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
