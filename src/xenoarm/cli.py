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
    execution.add_argument("--progress", action=argparse.BooleanOptionalAction, default=None)
    execution.add_argument("--wandb", choices=("disabled", "offline", "online"))
    execution.add_argument("--wandb-project")
    execution.add_argument("--wandb-entity")
    execution.add_argument("--video", action=argparse.BooleanOptionalAction, default=None)
    execution.add_argument("--video-episodes", type=int)
    execution.add_argument("--video-stride", type=int)
    report = sub.add_parser("report", help="Rebuild the local HTML/video/plot report")
    report.add_argument("directory")
    check = sub.add_parser("validate")
    check.add_argument("config")
    analysis = sub.add_parser("compare")
    analysis.add_argument("reference")
    analysis.add_argument("candidate")
    args = parser.parse_args()
    if args.command == "report":
        from .reporting import generate_report

        print(generate_report(args.directory))
    elif args.command == "compare":
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
        if args.progress is not None:
            config["progress"] = args.progress
        tracking = config.setdefault("tracking", {})
        for field, value in (
            ("mode", args.wandb),
            ("project", args.wandb_project),
            ("entity", args.wandb_entity),
        ):
            if value is not None:
                tracking[field] = value
        video = config.setdefault("visualization", {})
        for field, value in (
            ("enabled", args.video),
            ("max_episodes", args.video_episodes),
            ("frame_stride", args.video_stride),
        ):
            if value is not None:
                video[field] = value
        validate(config)
        run(config, args.output)
