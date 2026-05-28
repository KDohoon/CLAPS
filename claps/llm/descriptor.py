"""
Descriptor LLM (L_Describe): time-series x_t -> natural-language summary PB_t (Eq. 5).

Input  : a directory of per-scene prompt files (``*_prompt.txt``), each containing the
         downsampled sensor values for one scene serialized as '|'-delimited text.
Output : a JSON file of {key, summary} per scene, plus a human-readable report
         (scenarioN.txt) consumed by the cognitive pipeline as the internal cue IC_t.
"""

import os
import glob
import time
import json
from pathlib import Path

from claps.llm.client import LLMClient
from claps.llm import prompts
from claps.llm.user_prompts import build_descriptor_prompt


def run_descriptor(prompt_dir, out_dir, scenario_name, client=None,
                   sleep_between_calls=1.0):
    """
    Generate PB_t summaries for every ``*_prompt.txt`` in ``prompt_dir``.

    Writes:
      {out_dir}/{scenario_name}_results.json  -- [{key, status, response}, ...]
      {out_dir}/{scenario_name}.txt           -- readable report (cognitive-pipeline input)
    """
    client = client or LLMClient(temperature=0.2)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_files = sorted(glob.glob(os.path.join(prompt_dir, "*_prompt.txt")))
    if not prompt_files:
        print(f"[{scenario_name}] no *_prompt.txt found in {prompt_dir}")
        return

    results_path = out_dir / f"{scenario_name}_results.json"
    results, done = [], set()
    if results_path.exists():  # resume
        results = json.loads(results_path.read_text(encoding="utf-8"))
        done = {r["key"] for r in results if r.get("status") == "success"}
        print(f"[{scenario_name}] resuming, {len(done)} already done")

    for i, prompt_file in enumerate(prompt_files, 1):
        scene_key = Path(prompt_file).stem
        if scene_key in done:
            continue
        ts_data = Path(prompt_file).read_text(encoding="utf-8")
        text = client.chat(prompts.DESCRIPTOR_SYSTEM_PROMPT,
                           build_descriptor_prompt(ts_data),
                           label=f"{scenario_name}/{scene_key}")
        status = "success" if text else "error"
        results.append({"key": scene_key, "status": status, "response": text})
        print(f"[{scenario_name}] [{i}/{len(prompt_files)}] {scene_key} {status}")
        if i % 5 == 0 or i == len(prompt_files):
            results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        time.sleep(sleep_between_calls)

    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    # Readable report consumed by data.build_time_series_map
    report = out_dir / f"{scenario_name}.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write("VR 경험 분석 보고서\n" + "=" * 80 + "\n")
        f.write(f"시나리오: {scenario_name}\n총 결과: {len(results)}\n" + "=" * 80 + "\n\n")
        for r in results:
            if r.get("status") != "success":
                continue
            f.write("\n" + "=" * 80 + f"\nScene: {r['key']}\n" + "=" * 80 + "\n\n")
            f.write((r["response"] or "").strip() + "\n\n")
    print(f"[{scenario_name}] wrote {report}")
