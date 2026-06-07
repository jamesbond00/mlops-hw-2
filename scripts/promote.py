"""scripts/promote.py — promote MLflow Registry aliases with an audit log.

YOUR TASK (see tasks/task2.md): implement the four subcommand functions.
The argparse scaffolding below is wired so each cmd_* receives an `args`
namespace already parsed. See `_build_parser` for what's on `args` per
subcommand, and tasks/task2.md "Behavioral specs" for what each function
must do.

Versions are identified by their `config_id` tag (e.g., "v6"), NOT by
MLflow's integer version numbers. Resolution must be unique — if the
config_id matches zero or multiple registered versions, the CLI errors
out and forces the operator to disambiguate via the MLflow UI.

Successful `set` and `rollback` operations append a JSON event to
LOG_FILE (promotion-log.jsonl at repo root). `rollback` consults the
log to find the previous alias target.

Subcommands:
  set <alias> <config_id>   move alias, append `set` event to the log
  show <alias>              print current target + tags + key metrics
  list                      print all aliases on the registered model
  rollback <alias>          move alias back per the audit log, append
                            `rollback` event
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException, RestException
from mlflow.tracking import MlflowClient

from src.config import get_settings

settings = get_settings()
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflowClient: MlflowClient = MlflowClient()

REGISTERED_MODEL_NAME = "travel-assistant"
LOG_FILE = Path(__file__).resolve().parent.parent / "promotion-log.jsonl"


def _find_version_by_config_id(name: str, config_id: str):
    results = list(
        mlflowClient.search_model_versions(
            f"name = '{name}' AND tags.config_id = '{config_id}'"
        )
    )

    if not results:
        print(f"error: no version found with config_id={config_id}")
        sys.exit(1)

    results.sort(key=lambda version: int(version.version))
    if len(results) > 1:
        version_numbers = [int(version.version) for version in results]
        latest = results[-1]
        print(
            f"warning: multiple versions match config_id={config_id} "
            f"(MLflow versions {version_numbers}); using latest ({latest.version})"
        )
        return latest

    return results[0]

# Look up what args.alias currently points at 
def _current_alias_config_id(name: str, alias: str) -> str:
    try:
        current = mlflowClient.get_model_version_by_alias(name=name, alias=alias)
    except (MlflowException, RestException):
        return ""
    return (current.tags or {}).get("config_id", "")


def _get_alias_target(name: str, alias: str):
    try:
        return mlflowClient.get_model_version_by_alias(name=name, alias=alias)
    except (MlflowException, RestException):
        print(f"error: alias {alias} is not set")
        sys.exit(1)


def _latest_log_event_for_alias(alias: str) -> dict | None:
    if not LOG_FILE.exists():
        return None

    with LOG_FILE.open("r", encoding="utf-8") as log_file:
        lines = log_file.readlines()

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        if event.get("alias") == alias:
            return event

    return None


def _append_promotion_log(
    alias: str, from_config_id: str, to_config_id: str, op: str
) -> None:
    event = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "alias": alias,
        "from": from_config_id,
        "to": to_config_id,
        "op": op,
    }

    with LOG_FILE.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(event) + "\n")


def cmd_set(args: argparse.Namespace) -> None:
    """args.alias: str, args.config_id: str. See tasks/task2.md → cmd_set."""
    
    # find MLFlow versions by config_id
    target = _find_version_by_config_id(args.name, args.config_id)
    
    current_config_id = _current_alias_config_id(args.name, args.alias)

    mlflowClient.set_registered_model_alias(
        name=args.name,
        alias=args.alias,
        version=target.version,
    )

    _append_promotion_log(args.alias, current_config_id, args.config_id, "set")

    previous = current_config_id or "(unset)"
    print(f"{args.alias}: {previous} → {args.config_id}")


def cmd_show(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_show."""
    target = _get_alias_target(args.name, args.alias)
    tags = target.tags or {}
    metrics = mlflowClient.get_run(target.run_id).data.metrics

    print(f"{args.name} @ {args.alias}")

    config_id = tags.get("config_id", "")
    if config_id:
        print(f"  config_id: {config_id}")

    for key in sorted(k for k in tags if k != "config_id"):
        print(f"  {key}: {tags[key]}")

    metric_keys = ("accuracy_overall", "verdict_rate_leaked", "total_cost_usd")
    for key in metric_keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if key == "total_cost_usd":
            rendered = f"${value:.2f}"
        else:
            rendered = f"{value:.3f}".rstrip("0").rstrip(".")
        print(f"  {key}: {rendered}")


def cmd_list(args: argparse.Namespace) -> None:
    """No args. See tasks/task2.md → cmd_list."""

    # Get the registered model from MLflow.
    # args.name comes from the global CLI option --name, defaulting to travel-assistant.
    model = mlflowClient.get_registered_model(args.name)

    # Read aliases from the registered model:
    # MLflow returns this as a mapping like:
    # {
    #     "production": "12",
    #     "staging": "14",
    # }
    aliases = model.aliases or {}

    if not aliases:
        print("no aliases set")
        return

    for alias in sorted(aliases):
        target = mlflowClient.get_model_version_by_alias(args.name, alias)
        config_id = (target.tags or {}).get("config_id", "")
        print(f"{alias} -> {config_id}")


def cmd_rollback(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_rollback."""
    try:
        current = mlflowClient.get_model_version_by_alias(
            name=args.name,
            alias=args.alias,
        )
    except (MlflowException, RestException):
        print("nothing to roll back")
        sys.exit(1)

    current_config_id = (current.tags or {}).get("config_id", "")
    last_event = _latest_log_event_for_alias(args.alias)

    if last_event is None:
        print(f"no promotion history for alias {args.alias}")
        sys.exit(1)

    if last_event["op"] == "rollback":
        print(f"{args.alias} was just rolled back; no further history to walk back to")
        sys.exit(1)

    if last_event["from"] == "":
        print(f"{args.alias} has no previous target (first promotion ever)")
        sys.exit(1)

    # 4. Normal rollback
    previous_config_id = last_event["from"]
    target = _find_version_by_config_id(args.name, previous_config_id)

    mlflowClient.set_registered_model_alias(
        name=args.name,
        alias=args.alias,
        version=target.version,
    )

    _append_promotion_log(
        args.alias,
        current_config_id,
        previous_config_id,
        "rollback",
    )

    print(f"{args.alias}: {current_config_id} → {previous_config_id} (rolled back)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=REGISTERED_MODEL_NAME,
        help=f"Registered model name (default: {REGISTERED_MODEL_NAME})",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser(
        "set", help="Move an alias to a version (by config_id), append a set event"
    )
    p_set.add_argument("alias", help="Alias to assign (e.g., 'production')")
    p_set.add_argument(
        "config_id",
        help="Config identifier (e.g., 'v6') — resolved via the config_id tag on registered versions",
    )
    p_set.set_defaults(func=cmd_set)

    p_show = sub.add_parser("show", help="Show which version an alias points at")
    p_show.add_argument("alias")
    p_show.set_defaults(func=cmd_show)

    p_list = sub.add_parser("list", help="List all aliases on the registered model")
    p_list.set_defaults(func=cmd_list)

    p_rollback = sub.add_parser(
        "rollback",
        help="Move an alias back to its previous target per the audit log",
    )
    p_rollback.add_argument("alias")
    p_rollback.set_defaults(func=cmd_rollback)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.func(args)
    except NotImplementedError as exc:
        print(f"NOT IMPLEMENTED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
