"""
Recurrent cognitive-symptom generation pipeline (CLAPS, paper Section 2.1-2.2).

Per participant x scenario:
  1. Initializer L_Init                : (M_Long, S_init) -> (MR_0, AR)
  2. For each scene t = 1..N (recurrent):
       Updater   L_Update   : (MR_{t-1}, AR, S_t, IC_{t-1}) -> (MR_t, AES_t)
       Appraiser L_Appraise : (MR_t, AES_t)                 -> (Prob_t, Cons_t)
       Generator L_Generate : (Prob_t, Cons_t)              -> CS_t

The internal cue carried to the next scene is IC_t = (PB_t, CS_t), where PB_t is the
Descriptor LLM summary (precomputed; see descriptor.py) and CS_t is the cognitive
symptom from the Generator. For scene 1, IC_0 = empty and MR_1 = MR_0.

Outputs (JSON):
  initializer.json        -- MR_0 / AR per (pid, scenario)
  cognitive_trace.json    -- full per-scene records of every LLM stage
  cognitive_symptoms.json -- CS_t per scene; the file consumed by the CAT module
"""

import json
from pathlib import Path

from claps.llm.client import LLMClient, extract_json
from claps.llm import prompts
from claps.llm.user_prompts import (
    build_initializer_prompt, build_updater_prompt,
    build_appraiser_prompt, build_generator_prompt,
)
from claps.llm.data import get_scene


def _save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_initializer(client, participants, scenes, scenario_keys=("S1", "S2")):
    """L_Init for every (participant, scenario). Returns {(pid, sk): {MR_0, AR}}."""
    init_map = {}
    for p in participants:
        for sk in scenario_keys:
            init_scene = get_scene(scenes, sk, 1)
            text = client.chat(prompts.INITIALIZER_SYSTEM_PROMPT,
                               build_initializer_prompt(p, init_scene),
                               label=f"Init|P{p['pid']:02d}|{sk}")
            parsed = extract_json(text) or {}
            # accept either key spelling for the attention-allocation field
            mr = parsed.get("mental representation")
            ar = (parsed.get("Preferential Allocation of Attentional Resources")
                  or parsed.get("preferential allocation"))
            if mr and ar:
                init_map[(p["pid"], sk)] = {"mental representation": mr,
                                            "preferential allocation": ar}
    return init_map


def run_pipeline(participants, scenes, scene_counts, ts_map, out_dir,
                 client=None, scenario_keys=("S1", "S2")):
    """
    Run the full recurrent pipeline and write the three output JSON files.

    participants : list of self-report dicts (see data.load_participants)
    scenes       : scene profiles (see data.load_scene_profiles)
    scene_counts : {"S1": n1, "S2": n2}
    ts_map       : {(pid, sk, scene): PB_t}  (Descriptor outputs; may be empty)
    """
    client = client or LLMClient(temperature=0.1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_map = run_initializer(client, participants, scenes, scenario_keys)
    _save(out_dir / "initializer.json",
          [{"pid": pid, "scenario": sk, **v} for (pid, sk), v in init_map.items()])

    mr_map, cs_map = {}, {}          # (pid, sk, scene) -> text
    trace, cs_records = [], []        # output accumulators
    max_scene = max(scene_counts.values())

    for scene_num in range(1, max_scene + 1):
        for (pid, sk), base in init_map.items():
            if scene_num > scene_counts[sk]:
                continue

            # ---- internal cue IC_{t-1} ----
            if scene_num == 1:
                prev_mr = base["mental representation"]
                prev_cs = prev_pb = None
            else:
                prev_mr = mr_map.get((pid, sk, scene_num - 1))
                prev_cs = cs_map.get((pid, sk, scene_num - 1))
                prev_pb = ts_map.get((pid, sk, scene_num - 1))
                if prev_mr is None:        # previous scene failed; skip chain
                    continue

            label = f"P{pid:02d}|{sk}|s{scene_num}"

            # ---- Updater ----
            upd = extract_json(client.chat(
                prompts.UPDATER_SYSTEM_PROMPT,
                build_updater_prompt({
                    "previous mental representation": prev_mr,
                    "preferential allocation": base["preferential allocation"],
                    "previous cognitive symptoms of anxiety": prev_cs,
                    "previous physiological and behavioral summary": prev_pb,
                    "current_scene": get_scene(scenes, sk, scene_num),
                }),
                label=f"Update|{label}")) or {}
            mr = upd.get("mental representation")
            aes = upd.get("appraisal of audience's expected standard")
            if not (mr and aes):
                continue
            mr_map[(pid, sk, scene_num)] = mr

            # ---- Appraiser ----
            apr = extract_json(client.chat(
                prompts.APPRAISER_SYSTEM_PROMPT,
                build_appraiser_prompt({
                    "mental representation": mr,
                    "appraisal of audience's expected standard": aes,
                }),
                label=f"Appraise|{label}")) or {}
            prob = apr.get("judgement of probability of negative evaluation from audience")
            cons = apr.get("judgement of consequence of negative evaluation from audience")
            if not (prob and cons):
                continue

            # ---- Generator ----
            gen = extract_json(client.chat(
                prompts.GENERATOR_SYSTEM_PROMPT,
                build_generator_prompt({
                    "judgement of probability of negative evaluation from audience": prob,
                    "judgement of consequence of negative evaluation from audience": cons,
                }),
                label=f"Generate|{label}")) or {}
            cs = gen.get("cognitive symptoms of anxiety")
            if not cs:
                continue
            cs_map[(pid, sk, scene_num)] = cs

            trace.append({
                "pid": pid, "scenario": sk, "scene_number": scene_num,
                "mental representation": mr,
                "appraisal of audience's expected standard": aes,
                "judgement of probability of negative evaluation from audience": prob,
                "judgement of consequence of negative evaluation from audience": cons,
                "cognitive symptoms of anxiety": cs,
            })
            cs_records.append({
                "pid": pid, "scenario": sk, "scene_number": scene_num,
                "parsed": {"cognitive symptoms of anxiety": cs},
                "status": "success",
            })

        # incremental save after each scene
        _save(out_dir / "cognitive_trace.json", trace)
        _save(out_dir / "cognitive_symptoms.json", cs_records)
        print(f"scene {scene_num}/{max_scene}: {len(cs_records)} cognitive symptoms so far")

    print(f"done. {len(cs_records)} scene-level cognitive symptoms -> {out_dir}")
    return cs_records
