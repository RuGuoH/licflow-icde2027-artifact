"""Construct a structured entry from a license text supplied by the user."""

import argparse
import os
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("license_id")
    parser.add_argument("text", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="deepseek", help="Provider preset or API model name")
    parser.add_argument("--method", choices=("qa", "legacy"), default="qa")
    args = parser.parse_args()

    source = args.text.read_text(encoding="utf-8").strip()
    if not source:
        parser.error("the license text file is empty")
    from scripts.llm_license_helper import LLMLicenseHelper

    os.environ["LICENSE_MODELING_PIPELINE"] = args.method
    helper = LLMLicenseHelper(api_key=os.getenv("LICENSE_LLM_API_KEY"), model=args.model)
    if not helper.api_key:
        parser.error("set the API credential for the selected model")
    helper.license_cache[args.license_id] = source
    entry = helper.get_license_data(args.license_id)
    if entry is None:
        raise SystemExit("entry construction failed")
    args.output.write_text(yaml.safe_dump(entry, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    main()
