"""
src/metrics.py
Multi-tiered evaluation metrics: BLEU-4, CodeBERTScore wrapper, and GPT-4o-mini rubric judge.
"""

from typing import Dict, List, Any
import sacrebleu


def compute_lexical_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes standard surface-level lexical metrics (BLEU-4 and Sentence BLEU).
    """
    # SacreBLEU expects references as a list of lists of strings: [[ref1_v1, ...], [ref2_v1, ...]]
    formatted_refs = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(predictions, formatted_refs)

    return {
        "bleu4": round(bleu.score, 3),
    }


def compute_codebert_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes token-level semantic similarity using CodeBERT embeddings.
    """
    try:
        from code_bert_score import score
        P, R, F1 = score(predictions, references, lang="python", rescale_with_baseline=True)
        return {
            "codebert_precision": round(float(P.mean()), 4),
            "codebert_recall": round(float(R.mean()), 4),
            "codebert_f1": round(float(F1.mean()), 4),
        }
    except ImportError:
        return {"codebert_f1": -1.0, "error": "code-bert-score package not installed"}


# System and user prompt templates for GPT-4o-mini rubric judging
JUDGE_SYSTEM_PROMPT = """You are an expert software engineering judge assessing the factual accuracy and semantic fidelity of generated issue reports reconstructed from code changes.
Grade the candidate issue description against the reference issue description strictly according to the provided rubric."""

JUDGE_USER_RUBRIC_TEMPLATE = """Evaluate how accurately the candidate issue report reconstructs the original bug report.

### Ground Truth Problem Statement:
{reference_issue}

### Candidate Generated Issue Statement:
{predicted_issue}

### Scoring Criteria:
1. Symptom Fidelity (1-5): Does the candidate correctly describe the user-observable failure, unexpected exception, or incorrect behavior?
   - 1: Completely missing or wrong symptoms.
   - 3: Partially identified (vague symptom or minor mismatch).
   - 5: Exactly identifies the failure symptoms and conditions.

2. Root Cause Precision (1-5): Does the candidate accurately isolate the faulty logic or component?
   - 1: Fabricated or unrelated root cause.
   - 3: General area correct, but imprecise diagnosis.
   - 5: Pinpoints the exact bug mechanism.

3. Hallucination Penalty (0-1): Does the candidate fabricate non-existent APIs, error traces, or external libraries?
   - 0.0: No hallucinations; strictly grounded.
   - 0.5: Minor speculative details.
   - 1.0: Severe fabrications that mislead debugging.

Respond strictly in valid JSON format with this exact schema:
{{
  "symptom_fidelity": <int between 1 and 5>,
  "root_cause_precision": <int between 1 and 5>,
  "hallucination_penalty": <float between 0.0 and 1.0>,
  "composite_score": <float: (symptom_fidelity + root_cause_precision) / 2 * (1 - hallucination_penalty * 0.5)>,
  "reasoning": "<concise explanation>"
}}
"""


def format_judge_prompt(reference: str, prediction: str) -> Dict[str, str]:
    """
    Generates structured API messages ready for the GPT-4o-mini client.
    """
    return {
        "system": JUDGE_SYSTEM_PROMPT,
        "user": JUDGE_USER_RUBRIC_TEMPLATE.format(
            reference_issue=reference.strip(),
            predicted_issue=prediction.strip()
        ),
    }