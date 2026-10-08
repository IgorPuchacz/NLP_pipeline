"""
src/context_builder.py
Logic for unified diff parsing, repository scope extraction, and C0–C5 prompt assembly.
"""

import ast
import os
import re
from typing import Dict, List, Optional, Tuple, Any

# Regex to match unified diff hunk headers: @@ -start,len +start,len @@
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_DIFF_FILE_RE = re.compile(r"^(?:---|\+\+\+) (?:[ab]/)?(.*)$")


def parse_patch_hunks(patch_text: str) -> List[Dict[str, Any]]:
    """
    Parses a unified diff patch string into modified files and their hunk line ranges.
    Returns a list of dicts: {'file_path': str, 'start_line': int, 'line_count': int, 'hunk_text': str}
    """
    if not isinstance(patch_text, str) or not patch_text.strip():
        return []

    hunks = []
    current_file = None
    current_hunk = None

    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            match = _DIFF_FILE_RE.match(line)
            if match:
                current_file = match.group(1).strip()
                if current_file.startswith("b/"):
                    current_file = current_file[2:]
            continue

        hunk_match = _HUNK_HEADER_RE.match(line)
        if hunk_match and current_file:
            if current_hunk:
                hunks.append(current_hunk)

            start_line = int(hunk_match.group(2))
            line_count = int(hunk_match.group(3)) if hunk_match.group(3) else 1
            current_hunk = {
                "file_path": current_file,
                "start_line": start_line,
                "line_count": line_count,
                "lines": [line],
            }
        elif current_hunk is not None:
            current_hunk["lines"].append(line)

    if current_hunk:
        hunks.append(current_hunk)

    for h in hunks:
        h["hunk_text"] = "\n".join(h.pop("lines"))

    return hunks


def extract_ast_scope(source_code: str, target_line: int, max_tokens: int = 512) -> str:
    """
    Extracts the innermost enclosing function or class definition (K_small)
    around a target line number using Python's built-in AST parser.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        # Fallback to local line window if code does not parse cleanly
        lines = source_code.splitlines()
        start = max(0, target_line - 15)
        end = min(len(lines), target_line + 15)
        return "\n".join(lines[start:end])

    best_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                if node.lineno <= target_line <= node.end_lineno:
                    # Prefer narrower (innermost) scopes
                    if best_node is None or (node.end_lineno - node.lineno < best_node.end_lineno - best_node.lineno):
                        best_node = node

    lines = source_code.splitlines()
    if best_node and hasattr(best_node, "lineno") and hasattr(best_node, "end_lineno"):
        scope_lines = lines[best_node.lineno - 1 : best_node.end_lineno]
        extracted = "\n".join(scope_lines)
    else:
        # Fallback to 30 surrounding lines if no enclosing AST scope found
        start = max(0, target_line - 15)
        end = min(len(lines), target_line + 15)
        extracted = "\n".join(lines[start:end])

    # Truncate rough character approximation if scope exceeds max_tokens (~4 chars/token)
    char_limit = max_tokens * 4
    if len(extracted) > char_limit:
        extracted = extracted[:char_limit] + "\n# [... scope truncated ...]"
    return extracted


def extract_file_window(source_code: str, target_line: int, max_tokens: int = 4096) -> str:
    """
    Extracts broad surrounding file context (K_large) centered on the target line,
    bounded by the token ceiling.
    """
    lines = source_code.splitlines()
    total_lines = len(lines)
    max_lines = max_tokens // 8  # heuristic: ~8 tokens per code line

    if total_lines <= max_lines:
        return source_code

    half = max_lines // 2
    start = max(0, target_line - half)
    end = min(total_lines, start + max_lines)
    if end == total_lines:
        start = max(0, end - max_lines)

    selected = lines[start:end]
    return f"# [... lines 1 to {start} omitted ...]\n" + "\n".join(selected) + f"\n# [... lines {end} to {total_lines} omitted ...]"


def build_context(
    instance: Dict[str, Any],
    config: str,
    repo_root: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    max_total_tokens: int = 8192,
) -> str:
    """
    Assembles the input prompt X_c for configurations C0 through C5:
      C0: Diff Only
      C1: Diff + Tests
      C2: Diff + Local AST Scope
      C3: Diff + Broad File Context
      C4: Diff + Tests + Local AST Scope
      C5: Diff + Tests + Broad File Context
    """
    config = config.upper()
    if config not in {"C0", "C1", "C2", "C3", "C4", "C5"}:
        raise ValueError(f"Invalid configuration '{config}'. Choose from C0 through C5.")

    patch = instance.get("patch", "").strip()
    test_patch = instance.get("test_patch", "").strip()
    hunks = parse_patch_hunks(patch)

    sections = [
        "### Instructions:",
        "Analyze the following code modifications and reverse-engineer the original issue report.",
        "Infer the user-facing failure symptoms, unexpected behavior, and root problem that this change resolved.",
        "",
        "### Code Diff (Patch):",
        f"```diff\n{patch}\n```",
    ]

    # Include Unit Tests for C1, C4, C5
    if config in {"C1", "C4", "C5"} and test_patch:
        sections.extend([
            "",
            "### Relevant Unit Tests:",
            f"```python\n{test_patch}\n```",
        ])

    # Include Local AST Scope for C2, C4
    if config in {"C2", "C4"}:
        scope_snippets = []
        if repo_root and os.path.exists(repo_root):
            for h in hunks:
                full_path = os.path.join(repo_root, h["file_path"])
                if os.path.exists(full_path):
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                    snippet = extract_ast_scope(code, h["start_line"], max_tokens=512)
                    scope_snippets.append(f"File: {h['file_path']}\n{snippet}")
        if scope_snippets:
            sections.extend([
                "",
                "### Enclosing Function/Class Context (K_small):",
                "```python\n" + "\n\n".join(scope_snippets) + "\n```",
            ])
        else:
            # Fallback label when repo checkout is deferred
            sections.extend(["", "### Enclosing Function/Class Context (K_small):", "[Repository context extracted at checkout]"])

    # Include Broad File Context for C3, C5
    if config in {"C3", "C5"}:
        file_snippets = []
        if repo_root and os.path.exists(repo_root):
            for h in hunks:
                full_path = os.path.join(repo_root, h["file_path"])
                if os.path.exists(full_path):
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                    snippet = extract_file_window(code, h["start_line"], max_tokens=4096)
                    file_snippets.append(f"File: {h['file_path']}\n{snippet}")
        if file_snippets:
            sections.extend([
                "",
                "### Surrounding File Context (K_large):",
                "```python\n" + "\n\n".join(file_snippets) + "\n```",
            ])
        else:
            sections.extend(["", "### Surrounding File Context (K_large):", "[Broad file context extracted at checkout]"])

    sections.extend(["", "### Reconstructed Issue Description:"])
    prompt = "\n".join(sections)

    # Optional hard truncation safeguard
    if tokenizer is not None:
        tokens = tokenizer.encode(prompt, truncation=True, max_length=max_total_tokens)
        prompt = tokenizer.decode(tokens, skip_special_tokens=True)

    return prompt