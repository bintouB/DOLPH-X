# DOLPH-X: Understanding Opaque Student Profile Prediction using Explainable AI and Large Language Models

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Master 2 MIAGE](https://img.shields.io/badge/Master%202-MIAGE%20IKSEM-navy)](https://www.pantheonsorbonne.fr/)

> Master 2 MIAGE IKSEM — Université Paris 1 Panthéon-Sorbonne — 2025/2026
> Author: **Bintou BARADJI**
> Supervisor: Nourhène Ben Rabah

---

## Overview

**DOLPH-X** is a four-phase end-to-end pipeline that transforms raw LMS data into personalised, pedagogically grounded student feedback. It combines:

- **Machine Learning** — multi-class student profile classification
- **Explainable AI (XAI)** — multi-method explanation extraction
- **LLM-as-a-Judge** — automated XAI quality validation
- **Generative LLMs** — pedagogically constrained feedback generation

The pipeline was developed and evaluated on a dataset of 128 students enrolled in an online Python programming course at Université Paris 1 Panthéon-Sorbonne (cohort 2021-2022).

---

## Pipeline Architecture

```
LMS Data (128 students, 106 variables)
         │
         ▼
┌─────────────────────┐
│  Phase 1: ML        │  XGBoost F1=0.840 — 4-class E×P profiling
│  Modelling          │  Profiles: E+P+, E+P-, E-P+, E-P-
└────────┬────────────┘
         │ 13 representative instances
         ▼
┌─────────────────────┐
│  Phase 2: XAI       │  LIME · MC-LIME · SHAP · DiCE
│  Extraction         │  Applied on 3 models × 13 instances
└────────┬────────────┘
         │ Consolidated XAI table
         ▼
┌─────────────────────┐
│  Phase 3:           │  GPT-OSS 120B via Groq API
│  LLM-as-a-Judge     │  105 validated features · mean confidence 0.752
└────────┬────────────┘
         │ validated_features.csv
         ▼
┌─────────────────────┐
│  Phase 4: Feedback  │  Two-LLM sequential architecture
│  Generation         │  SDT · TPB · Growth Mindset · Hattie model
└─────────────────────┘
         │
         ▼
Personalised, pedagogically grounded student feedback
```

---

## Student Profiles

| Profile | Meaning | Proportion |
|---------|---------|------------|
| E+P+ | Engaged and high-performing | 25.0% |
| E+P- | Engaged but low-performing | 14.8% |
| E-P+ | Disengaged but high-performing | 35.2% |
| E-P- | Disengaged and low-performing | 25.0% |

The **E-P+ profile** (35.2% of students) is invisible in any binary pass/fail framework — one of the key motivations for the four-class formulation.

---

## Repository Structure

```
DOLPH-X/
├── README.md
├── MLEvaluator.py              # ML training and evaluation class (11 models)
├── XAIEvaluator.py             # Base class shared by all XAI evaluators
├── 01_phase1_ML.ipynb          # Phase 1: Preprocessing + ML modelling
├── 02a_phase2_LIME.ipynb       # Phase 2a: LIME explanations
├── 02b_phase2_MC_LIME.ipynb    # Phase 2b: MC-LIME multi-class explanations
├── 02c_phase2_SHAP.ipynb       # Phase 2c: SHAP explanations
├── 02d_phase2_DiCE.ipynb       # Phase 2d: DiCE counterfactuals
├── 03_phase3_llm_judge.ipynb   # Phase 3: LLM-as-a-Judge validation
├── 04_phase4_feedback.ipynb    # Phase 4: Personalised feedback generation
└── outputs/
    └── .gitkeep                # Output directory (generated files not tracked)
```

---

## Setup

### Requirements

```bash
conda create -n dolphx python=3.9
conda activate dolphx
```

```bash
pip install numpy pandas matplotlib seaborn scikit-learn==1.1.3
pip install xgboost imbalanced-learn==0.10.1
pip install shap==0.41.0
pip install dice-ml
pip install groq
```

> **Apple Silicon (M1/M2):** `brew install libomp` before installing XGBoost.

### API Key

Phase 3 and Phase 4 require a [Groq API key](https://console.groq.com) (free tier).

Set it in the first cell of `03_phase3_llm_judge.ipynb` and `04_phase4_feedback.ipynb`:

```python
os.environ['GROQ_API_KEY'] = 'gsk_...'
```

---

## Running the Pipeline

Run notebooks in order:

| Notebook | Generates |
|----------|-----------|
| `01_phase1_ML.ipynb` | `outputs/evaluator.pkl` |
| `02a_phase2_LIME.ipynb` | `outputs/lime_explanations.csv` |
| `02b_phase2_MC_LIME.ipynb` | `outputs/mclime_explanations.csv` |
| `02c_phase2_SHAP.ipynb` | `outputs/shap_explanations.csv` |
| `02d_phase2_DiCE.ipynb` | `outputs/dice_explanations.csv` |
| `03_phase3_llm_judge.ipynb` | `outputs/validated_features.csv` |
| `04_phase4_feedback.ipynb` | `outputs/feedback_final.csv` |

> **Note:** Run Phase 3 and Phase 4 on separate days to avoid exceeding the Groq free tier limit (200,000 tokens/day for GPT-OSS 120B).

---

## Key Results

| Phase | Key result |
|-------|-----------|
| Phase 1 | XGBoost F1=0.840, AUC=0.971 |
| Phase 2 | SHAP fidelity: 1.000 (DT, RF) · LIME fidelity: 0.000–0.068 |
| Phase 3 | 105 validated features · mean confidence 0.752 · 98.1% actionable |
| Phase 4 | 4/4 feedbacks complete · 183–205 words · all Hattie constraints satisfied |

---

## Important Notes

### Reproducibility

The pipeline uses `random_state=42` throughout. All results are reproducible provided the same `evaluator.pkl` is used. **Do not re-run Phase 1** if you want to reproduce the exact XAI and feedback results — load the existing `evaluator.pkl` instead.

### Known Limitations

- **DiCE**: Failed to generate valid counterfactuals due to large feature-space distance and StandardScaler normalisation constraints. `dice_explanations.csv` is empty.
- **Groq quota**: 200,000 tokens/day limit for GPT-OSS 120B. Phase 3 and Phase 4 must run on separate days.
- **Dataset**: Not included in this repository (student data — confidentiality). Contact the author if needed.
- **evaluator.pkl**: Generated with Python 3.9 and scikit-learn 1.1.3. Not compatible with other versions.

---

## Pedagogical Theories (Phase 4)

The feedback generation stage is explicitly constrained by four social science theories:

| Theory | Authors | Constraint in the prompt |
|--------|---------|--------------------------|
| Self-Determination Theory | Deci & Ryan (2000) | Autonomy-supportive language only |
| Theory of Planned Behavior | Ajzen (1991) | Specific, achievable, time-bounded actions |
| Growth Mindset | Dweck (2006) | No fixed-mindset language |
| Hattie Feedback Model | Hattie & Timperley (2007) | Feed Up / Feed Back / Feed Forward structure |

---

## Citation

If you use DOLPH-X in your work, please cite:

```
@mastersthesis{baradji2026dolphx,
  author    = {Bintou Baradji},
  title     = {Understanding Opaque Student Profile Prediction using
               Explainable AI and Large Language Models},
  school    = {Universite Paris 1 Pantheon-Sorbonne},
  year      = {2026},
  type      = {Master 2 MIAGE IKSEM Thesis}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Master 2 MIAGE IKSEM · Université Paris 1 Panthéon-Sorbonne · 2025–2026*
