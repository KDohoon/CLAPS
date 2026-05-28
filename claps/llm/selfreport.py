"""
Self-report scoring for the Initializer LLM's long-term memory (M_Long).

SPM = Self-Presentation Motivation, SPE = Self-Presentation Expectancy (7-item,
7-point Likert each). The social-anxiety proxy used in the paper is the discrepancy

    Diff = SPM_norm * (1 - SPE_norm)

where each 1..7 average is min-max normalized to [0, 1] via (x - 1) / 6.
"""

SPM_ITEMS_TEXT = [
    "나는 일을 하는 내 모양새에 관심이 많다",
    "나는 다른 사람에게 나를 제시하는 방식에 관심이 많다",
    "나는 내가 어떤 모습으로 보일지에 대해 많이 의식한다",
    "일상적으로 나는 좋은 인상을 주는 것에 대해 걱정을 많이 한다",
    "나는 집을 나서기 전에 꼭 거울을 본다",
    "나는 다른 사람들이 나를 어떻게 생각하는가에 대해 관심이 많다",
    "나는 일상적으로 나의 외모를 많이 의식한다",
]
SPE_ITEMS_TEXT = [
    "낯선 사람들과 처음 만나는 모임에서 인사를 나누고 자기소개를 하는 상황",
    "직장을 구하기 위한 면접시험에서 면접관과 만나는 상황",
    "모르는 사람과 대화를 나누는 상황",
    "대중 앞에서 혹은 상급자와 대화를 나누는 상황",
    "이성과 대화를 나누는 상황",
    "비난 받거나 창피를 당하는 상황",
    "짜증, 혐오, 불만에 대해 표현하는 상황",
]

# Theoretical range of Diff under [0, 1] normalization.
DIFF_THEORY_MIN = 0.0   # SPM=1 or SPE=7
DIFF_THEORY_MAX = 1.0   # SPM=7, SPE=1


def normalized_diff(spm_avg, spe_avg):
    """Return (spm_norm, spe_norm, diff) for 1..7 averages."""
    spm_norm = (spm_avg - 1) / 6
    spe_norm = (spe_avg - 1) / 6
    return spm_norm, spe_norm, spm_norm * (1 - spe_norm)


def compute_spm_spe_diff(participant):
    """participant: dict with 'spm_items', 'spe_items' (length-7 lists)."""
    spm_avg = sum(participant["spm_items"]) / 7
    spe_avg = sum(participant["spe_items"]) / 7
    spm_norm, spe_norm, diff = normalized_diff(spm_avg, spe_avg)
    return spm_avg, spe_avg, spm_norm, spe_norm, diff
