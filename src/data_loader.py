"""
src/data_loader.py
Dataset loading, filtering, and sanitization routines for SWE-bench instances.
"""

import re
from typing import Optional, Dict, Any, List
import pandas as pd
from datasets import load_dataset, Dataset

# Regex patterns for markdown, HTML, and media sanitization
_HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[.*?\]\(.*?\)")
_RAW_HTML_IMG_PATTERN = re.compile(r"<img[^>]*>", flags=re.IGNORECASE)
_BASE64_PATTERN = re.compile(r"data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+")
_EXCESS_WHITESPACE = re.compile(r"\n{3,}")


def _compress_tracebacks(text: str) -> str:
    """
    Linear O(N) scan that safely truncates intermediate frames in deep stack traces
    without triggering regular expression catastrophic backtracking.
    """
    if "Traceback (most recent call last):" not in text:
        return text

    lines = text.splitlines()
    result: List[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if "Traceback (most recent call last):" in line:
            result.append(line)
            i += 1
            tb_frames: List[str] = []

            # Gather traceback frame lines until we hit an unindented line (the error) or blank line
            while i < n:
                curr = lines[i]
                if curr.strip() == "":
                    i += 1
                    break
                # If unindented, it's typically the final exception statement (e.g. ValueError: ...)
                if not curr.startswith(" ") and not curr.startswith("\t"):
                    tb_frames.append(curr)
                    i += 1
                    break
                tb_frames.append(curr)
                i += 1

            # Truncate intermediate frames if the traceback is long
            if len(tb_frames) > 6:
                result.extend(tb_frames[:2])
                result.append("  [... intermediate stack frames truncated ...]")
                result.extend(tb_frames[-2:])
            else:
                result.extend(tb_frames)
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


def clean_problem_statement(text: str) -> str:
    """
    Sanitizes raw GitHub issue descriptions:
    1. Removes HTML comments and base64 image blobs.
    2. Removes markdown and HTML image tags.
    3. Collapses repetitive traceback frames linearly while preserving the exception signature.
    4. Normalizes extraneous whitespace.
    """
    if not isinstance(text, str):
        return ""

    # Remove hidden comments and image embeds
    cleaned = _HTML_COMMENT_PATTERN.sub("", text)
    cleaned = _BASE64_PATTERN.sub("", cleaned)
    cleaned = _MARKDOWN_IMAGE_PATTERN.sub("", cleaned)
    cleaned = _RAW_HTML_IMG_PATTERN.sub("", cleaned)

    # Compress deep stack traces safely
    cleaned = _compress_tracebacks(cleaned)

    # Normalize excessive newlines
    cleaned = _EXCESS_WHITESPACE.sub("\n\n", cleaned)
    return cleaned.strip()


def strip_test_patch_comments(patch_text: str) -> str:
    """
    Removes inline comments, comment-only lines, and docstrings from unified test patches
    to prevent trivial semantic leakage into context configurations.
    """
    if not isinstance(patch_text, str) or not patch_text.strip():
        return ""

    cleaned_lines = []
    in_multiline_docstring = False

    for line in patch_text.splitlines():
        # Keep diff headers untouched (@@, +++, ---, diff --git, etc.)
        if line.startswith(("diff --git", "index ", "--- ", "+++ ", "@@")):
            cleaned_lines.append(line)
            continue

        prefix = line[0] if line.startswith(("+", "-", " ")) else " "
        content = line[1:] if line.startswith(("+", "-", " ")) else line
        stripped_content = content.strip()

        # Handle docstrings (""" or ''')
        if '"""' in stripped_content or "'''" in stripped_content:
            quotes = '"""' if '"""' in stripped_content else "'''"
            if stripped_content.count(quotes) == 1:
                in_multiline_docstring = not in_multiline_docstring
                continue
            elif stripped_content.count(quotes) >= 2:
                continue

        if in_multiline_docstring:
            continue

        # Drop lines that are pure comments
        if stripped_content.startswith("#"):
            continue

        # Strip trailing inline comments safely
        if "#" in content:
            parts = re.split(r"(?<!['\"])#(?!['\"])", content, maxsplit=1)
            content = parts[0].rstrip()

        if content.strip() or prefix == " ":
            cleaned_lines.append(f"{prefix}{content}")

    return "\n".join(cleaned_lines)


def load_swebench_dataset(split: str = "verified", cache_dir: Optional[str] = None) -> Dataset:
    """
    Loads the requested SWE-bench split from Hugging Face.
    """
    repo_mapping = {
        "verified": ("princeton-nlp/SWE-bench_Verified", "test"),
        "lite": ("princeton-nlp/SWE-bench_Lite", "test"),
        "lite_train": ("princeton-nlp/SWE-bench_Lite", "train"),
        "full_train": ("princeton-nlp/SWE-bench", "train"),
    }

    if split not in repo_mapping:
        raise ValueError(f"Unknown split '{split}'. Available: {list(repo_mapping.keys())}")

    dataset_name, dataset_split = repo_mapping[split]
    print(f"Loading {dataset_name} ({dataset_split} split)...")
    return load_dataset(dataset_name, split=dataset_split, cache_dir=cache_dir)


def preprocess_swebench_dataset(
    dataset: Dataset,
    min_tokens: int = 20,
    tokenizer: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Cleans, sanitizes, and filters a SWE-bench Dataset into an analysis-ready pandas DataFrame.
    """
    records = []

    for item in dataset:
        raw_issue = item.get("problem_statement", "")
        clean_issue = clean_problem_statement(raw_issue)
        raw_test_patch = item.get("test_patch", "")
        clean_tests = strip_test_patch_comments(raw_test_patch)
        patch = item.get("patch", "")

        if tokenizer is not None:
            issue_len = len(tokenizer.encode(clean_issue, add_special_tokens=False))
            patch_len = len(tokenizer.encode(patch, add_special_tokens=False))
            test_len = len(tokenizer.encode(clean_tests, add_special_tokens=False))
        else:
            issue_len = len(clean_issue.split())
            patch_len = len(patch.split())
            test_len = len(clean_tests.split())

        # Filter out empty patches or overly brief problem descriptions (< min_tokens)
        if not patch.strip() or issue_len < min_tokens:
            continue

        records.append({
            "instance_id": item["instance_id"],
            "repo": item["repo"],
            "base_commit": item["base_commit"],
            "problem_statement": clean_issue,
            "raw_problem_statement": raw_issue,
            "patch": patch,
            "test_patch": clean_tests,
            "raw_test_patch": raw_test_patch,
            "token_count_issue": issue_len,
            "token_count_patch": patch_len,
            "token_count_test": test_len,
        })

    return pd.DataFrame(records)