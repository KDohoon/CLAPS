"""
Step 2 — Recurrent cognitive-symptom pipeline (Initializer -> Updater -> Appraiser -> Generator).

Produces outputs/cognitive_symptoms.json, the CS_t file consumed by the CAT module.

Usage:
    python scripts/run_cognitive_pipeline.py \
        --scene_profiles data/scene_profiles.json \
        --selfreport "data/VRST_dataset_all/유저스터디 결과정리_final.xlsx" \
        --descriptor_s1 outputs/descriptor/scenario1.txt \
        --descriptor_s2 outputs/descriptor/scenario2.txt \
        --dataset_root data/VRST_dataset_all \
        --out_dir outputs
"""
import argparse

from claps.llm.client import LLMClient
from claps.llm.data import (
    load_scene_profiles, load_participants, build_time_series_map,
)
from claps.llm.pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_profiles", required=True,
                    help="Per-scene external-cue / audience metadata JSON (see README for schema)")
    ap.add_argument("--selfreport", required=True,
                    help="Study spreadsheet with SPM/SPE per participant")
    ap.add_argument("--descriptor_s1", default=None, help="scenario1.txt (PB_t report)")
    ap.add_argument("--descriptor_s2", default=None, help="scenario2.txt (PB_t report)")
    ap.add_argument("--dataset_root", default=None,
                    help="Per-participant sensor dirs, used to join PB_t to pid")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--temperature", type=float, default=0.1)
    args = ap.parse_args()

    scenes, scene_counts = load_scene_profiles(args.scene_profiles)
    participants = load_participants(args.selfreport)

    ts_map = {}
    if args.descriptor_s1 and args.descriptor_s2 and args.dataset_root:
        ts_map = build_time_series_map(
            {"S1": args.descriptor_s1, "S2": args.descriptor_s2},
            args.dataset_root,
        )
    else:
        print("WARNING: no Descriptor reports given; running without internal cue PB_t.")

    client = LLMClient(temperature=args.temperature)
    run_pipeline(participants, scenes, scene_counts, ts_map, args.out_dir, client=client)


if __name__ == "__main__":
    main()
