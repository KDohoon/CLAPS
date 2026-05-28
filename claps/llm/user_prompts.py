"""
User-prompt builders for each CLAPS LLM component.

Each function turns a structured record into the user message that accompanies the
corresponding system prompt in ``prompts.py``.
"""

from claps.llm import prompts
from claps.llm.selfreport import (
    SPM_ITEMS_TEXT, SPE_ITEMS_TEXT, compute_spm_spe_diff,
    DIFF_THEORY_MIN, DIFF_THEORY_MAX,
)


# ---------------------------------------------------------------------------
# Descriptor LLM (L_Describe)
# ---------------------------------------------------------------------------
def build_descriptor_prompt(time_series_data):
    """``time_series_data``: per-scene sensor values serialized as '|'-delimited text."""
    return prompts.DESCRIPTOR_USER_TEMPLATE.format(
        description=prompts.DESCRIPTOR_FEATURE_DESCRIPTION,
        time_series_data=time_series_data,
    )


# ---------------------------------------------------------------------------
# Initializer LLM (L_Init)  — uses long-term memory (self-report) + initial scene
# ---------------------------------------------------------------------------
def build_initializer_prompt(participant, scene):
    """
    participant: dict with 'spm_items', 'spe_items' (7-item Likert lists).
    scene:       initial-scene profile dict (scene_profiles.json entry).
    """
    raw_scene = scene["raw_scene"]
    perceived_audience = scene["perceived_audience"]

    spm_text = "\n".join(
        f"- SPM-{i + 1} {SPM_ITEMS_TEXT[i]}: {participant['spm_items'][i]}"
        for i in range(7)
    )
    spe_text = "\n".join(
        f"- SPE-{i + 1} {SPE_ITEMS_TEXT[i]}: {participant['spe_items'][i]}"
        for i in range(7)
    )
    spm_avg, spe_avg, spm_norm, spe_norm, diff = compute_spm_spe_diff(participant)

    return f"""[입력 정보]
[SPM 문항 점수]
{spm_text}

[SPE 문항 점수]
{spe_text}

[SPM-SPE 차이 지표 ([0,1] 정규화)]
- SPM 7항목 평균: {spm_avg:.3f} (정규화: {spm_norm:.3f})
- SPE 7항목 평균: {spe_avg:.3f} (정규화: {spe_norm:.3f})
- SPM과 SPE 차이 = SPM_norm × (1 - SPE_norm): {diff:.3f}
- SPM과 SPE 차이 범위: [{DIFF_THEORY_MIN}, {DIFF_THEORY_MAX}]

[현재 장면 정보]
- 현재 장면 맥락: {raw_scene['situation']}
- 현재 과제: {raw_scene['user_task']}

[Perceived Audience]
- 청중 존재 여부: {perceived_audience['audience_presence']}
- 청중과의 친밀함: {perceived_audience['audience_familiarity']}
- 청중과의 관계: {perceived_audience['audience_relation']}
"""


# ---------------------------------------------------------------------------
# Updater LLM (L_Update)
# ---------------------------------------------------------------------------
def build_updater_prompt(record):
    """
    record keys:
      previous mental representation, preferential allocation,
      previous cognitive symptoms of anxiety, previous physiological and behavioral summary,
      current_scene
    """
    scene = record["current_scene"]
    raw_scene = scene["raw_scene"]
    perceived_audience = scene["perceived_audience"]
    current_scene_features = scene["current_scene_features"]
    internal_cog = record["previous cognitive symptoms of anxiety"] or "없음"
    internal_pb = record["previous physiological and behavioral summary"] or "없음"

    return f"""[입력 정보]

[직전 단계 출력]
- mental representation: {record['previous mental representation']}
- preferential allocation: {record['preferential allocation']}

[직전 내부 단서]
- 직전 장면에서 이어진 인지적 반응: {internal_cog}
- 직전 장면에서 이어진 생리적 반응과 행동적 반응: {internal_pb}

[현재 외부 단서 및 장면 정보]
- 현재 장면 맥락: {raw_scene['situation']}
- 현재 과제: {raw_scene['user_task']}
- 청중 수: {perceived_audience['audience_size']}
- 청중 관계: {perceived_audience['audience_relation']}
- 현재 장면의 과제 요구: {current_scene_features['task_demand']}
- 현재 장면의 언어적 단서: {current_scene_features['verbal_cues']}
- 현재 장면의 비언어적 단서: {current_scene_features['nonverbal_cues']}
"""


# ---------------------------------------------------------------------------
# Appraiser LLM (L_Appraise)
# ---------------------------------------------------------------------------
def build_appraiser_prompt(record):
    """record keys: mental representation, appraisal of audience's expected standard"""
    return f"""[입력 정보]
- 현재 mental representation: {record['mental representation']}
- 현재 appraisal of audience's expected standard: {record["appraisal of audience's expected standard"]}
"""


# ---------------------------------------------------------------------------
# Generator LLM (L_Generate)
# ---------------------------------------------------------------------------
def build_generator_prompt(record):
    """record keys: judgement of probability/consequence of negative evaluation from audience"""
    prob = record["judgement of probability of negative evaluation from audience"]
    cons = record["judgement of consequence of negative evaluation from audience"]
    return f"""[입력 정보]
- judgement of probability of negative evaluation from audience: {prob}
- judgement of consequence of negative evaluation from audience: {cons}
"""
