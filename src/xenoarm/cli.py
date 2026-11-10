import argparse
import json

from .analysis import compare
from .config import load_config, validate
from .runner import run


def main():
    parser = argparse.ArgumentParser(description="Light-WAM language-correction experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    execution = sub.add_parser("run")
    execution.add_argument("--config", required=True)
    execution.add_argument("--output", required=True)
    execution.add_argument("--mode", choices=("cancel", "drain"))
    execution.add_argument("--no-corrections", action="store_true")
    check = sub.add_parser("validate")
    check.add_argument("config")
    analysis = sub.add_parser("compare")
    analysis.add_argument("reference")
    analysis.add_argument("candidate")
    args = parser.parse_args()
    if args.command == "compare":
        print(json.dumps(compare(args.reference, args.candidate), indent=2))
    elif args.command == "validate":
        config = load_config(args.config)
        print(f"Valid: {config['name']} ({len(config['seeds'])} paired seeds)")
    else:
        config = load_config(args.config)
        if args.mode:
            config["mode"] = args.mode
        if args.no_corrections:
            config["corrections"] = []
        validate(config)
        run(config, args.output)
