"""
Data loading for the CLAPS LLM pipeline.

This wires three inputs into the recurrent cognitive-symptom pipeline:
  1. Scene profiles      -- per-scene external cue / audience metadata (scene_profiles.json)
  2. Self-report scores  -- SPM / SPE per participant (study spreadsheet)
  3. Descriptor outputs  -- per-scene physiological/behavioral summaries PB_t
                            (the scenarioN.txt reports produced by the Descriptor LLM)

The column indices and exclusion list below are specific to the VRST study
spreadsheet used in the paper; adapt ``load_participants`` to your own data layout.
"""

import re
import json
from pathlib import Path

import pandas as pd

# Map a short scenario key to the scenario id used inside scene_profiles.json
# and to the Descriptor report filename stem.
SCENE_KEY_TO_SOURCE = {"S1": "scenario1", "S2": "scenario2"}


def load_scene_profiles(path):
    with open(path, "r", encoding="utf-8") as f:
        scenes = json.load(f)
    scene_counts = {k: len(scenes[v]["scenes"]) for k, v in SCENE_KEY_TO_SOURCE.items()}
    return scenes, scene_counts


def get_scene(scenes, scenario_key, scene_number):
    """1-indexed scene lookup."""
    return scenes[SCENE_KEY_TO_SOURCE[scenario_key]]["scenes"][scene_number - 1]


def load_participants(excel_path, sheet_name="가공", exclude=(17, 24),
                      row_range=range(3, 37),
                      spm_cols=range(140, 147), spe_cols=range(147, 154),
                      interview_col=5, group_col=6, pid_col=0):
    """Read per-participant self-report records from the study spreadsheet."""
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    participants = []
    for i in row_range:
        pid = int(raw.iloc[i, pid_col])
        if pid in exclude:
            continue
        participants.append({
            "pid": pid,
            "spm_items": [int(raw.iloc[i, c]) for c in spm_cols],
            "spe_items": [int(raw.iloc[i, c]) for c in spe_cols],
            "interview_exp": str(raw.iloc[i, interview_col]),
            "group_exp": str(raw.iloc[i, group_col]),
        })
    return participants


# ---------------------------------------------------------------------------
# Descriptor output (PB_t) parsing
# ---------------------------------------------------------------------------
def _parse_scene_summary_file(path):
    """Parse one scenarioN.txt report into [{timestamp, scene_number, summary}, ...]."""
    text = Path(path).read_text(encoding="utf-8")
    header_pattern = re.compile(r"Scene:\s+([^\n]+)\n=+\n\n")
    matches = list(header_pattern.finditer(text))
    sections = []
    for idx, match in enumerate(matches):
        scene_key = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        summary = text[start:end].strip()
        m = re.match(r"(?P<timestamp>\d{8}_\d{6})_scene(?P<scene_number>\d+)_prompt", scene_key)
        if not m:
            continue
        sections.append({
            "timestamp": m.group("timestamp"),
            "scene_number": int(m.group("scene_number")),
            "summary": summary,
        })
    return sections


def build_time_series_map(summary_files, dataset_root,
                          csv_glob="*_data(head,e4,eye).csv"):
    """
    Map (pid, scenario_key, scene_number) -> Descriptor summary PB_t.

    summary_files : {"S1": <scenario1.txt>, "S2": <scenario2.txt>}
    dataset_root  : directory with one numbered subdir per participant, each holding
                    timestamped sensor CSVs whose names start with the recording
                    timestamp (used to join a recording session to a pid).
    """
    dataset_root = Path(dataset_root)

    # scenario key -> set of timestamps present in that Descriptor report
    scenario_timestamps = {}
    for sk, sp in summary_files.items():
        if not Path(sp).exists():
            continue
        scenario_timestamps[sk] = {s["timestamp"] for s in _parse_scene_summary_file(sp)}

    # timestamp -> pid, via the per-participant sensor CSV filenames
    timestamp_to_pid = {}
    for pid_dir in sorted(dataset_root.iterdir()):
        if not pid_dir.is_dir() or not pid_dir.name.isdigit():
            continue
        pid = int(pid_dir.name)
        for csv_file in sorted(pid_dir.glob(csv_glob)):
            ts = csv_file.name.split("_data")[0]
            for sk, ts_set in scenario_timestamps.items():
                if ts in ts_set:
                    timestamp_to_pid[(sk, ts)] = pid
                    break

    ts_map = {}
    for sk, sp in summary_files.items():
        if not Path(sp).exists():
            continue
        for section in _parse_scene_summary_file(sp):
            pid = timestamp_to_pid.get((sk, section["timestamp"]))
            if pid is not None:
                ts_map[(pid, sk, section["scene_number"])] = section["summary"]
    return ts_map
