# CLAPS

**Cognitive-behavioral LLM-Assisted Prediction of Social anxiety**

A multi-LLM framework that predicts social anxiety in Virtual Reality Exposure
Therapy (VRET) by (1) generating scene-level *cognitive symptoms* grounded in the
cognitive-behavioral theory of social anxiety, and (2) aligning those natural-language
symptoms with physiological/behavioral time-series via a Cross-modal Aligned
Transformer (CAT) to predict PEPQ-R scores.

This repository contains the **framework code only**. The VRET dataset, generated LLM
outputs, and trained checkpoints are not redistributed — see *Data layout* below.

## Framework → code map

The framework has five LLM components plus the CAT prediction module
(paper Section 2; equation numbers in parentheses).

| Paper component | Role | Code |
|---|---|---|
| **Descriptor** `L_Describe` | time-series `x_t` → NL summary `PB_t` (Eq. 5) | `claps/llm/descriptor.py` |
| **Initializer** `L_Init` | `(M_Long, S_init)` → `(MR_0, AR)` (Eq. 1) | `claps/llm/pipeline.py: run_initializer` |
| **Updater** `L_Update` | `(MR_{t-1}, AR, S_t, IC_{t-1})` → `(MR_t, AES_t)` (Eq. 2) | `claps/llm/pipeline.py: run_pipeline` |
| **Appraiser** `L_Appraise` | `(MR_t, AES_t)` → `(Prob_t, Cons_t)` (Eq. 3) | `claps/llm/pipeline.py: run_pipeline` |
| **Generator** `L_Generate` | `(Prob_t, Cons_t)` → cognitive symptom `CS_t` (Eq. 4) | `claps/llm/pipeline.py: run_pipeline` |
| **CAT** | align `CS_t` ⊗ time series → PEPQ-R (Eqs. 6–8) | `claps/cat/model.py` |

System prompts for the five LLMs live in `claps/llm/prompts.py`; the matching
user-prompt builders in `claps/llm/user_prompts.py`.

```
claps/
  llm/
    client.py        # OpenAI-compatible LLM client (API key from env) + JSON extraction
    prompts.py       # system prompts for all 5 LLMs
    user_prompts.py  # user-prompt builders
    selfreport.py    # SPM/SPE scoring (long-term memory M_Long)
    data.py          # scene profiles, self-report, Descriptor-output (PB_t) loaders
    descriptor.py    # Descriptor LLM runner
    pipeline.py      # recurrent Init → Update → Appraise → Generate
  cat/
    model.py         # Cross-modal Aligned Transformer (dual-path: scenario + scene)
    dataset.py       # sensor time series + cognitive text → PEPQ-R labels
    train.py         # k-fold training / evaluation
    layers/          # Transformer building blocks (patch embedding, attention)
    utils/
scripts/                      # CLI entry points
```
