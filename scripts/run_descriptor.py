"""
Step 1 — Descriptor LLM: turn per-scene sensor time series into PB_t summaries.

Usage:
    python scripts/run_descriptor.py --prompt_dir data/scene_prompts/scenario1 \
        --scenario scenario1 --out_dir outputs/descriptor
"""
import argparse

from claps.llm.client import LLMClient
from claps.llm.descriptor import run_descriptor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_dir", required=True,
                    help="Directory of per-scene *_prompt.txt files")
    ap.add_argument("--scenario", required=True, help="e.g. scenario1 / scenario2")
    ap.add_argument("--out_dir", default="outputs/descriptor")
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    client = LLMClient(temperature=args.temperature)
    run_descriptor(args.prompt_dir, args.out_dir, args.scenario, client=client)


if __name__ == "__main__":
    main()
