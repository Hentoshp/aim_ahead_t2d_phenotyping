import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Module 4 reporting pipeline (stub)")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    # TODO: implement reporting once upstream artifacts exist
    raise NotImplementedError("Module 4 reporting not yet implemented")


if __name__ == "__main__":
    main()
