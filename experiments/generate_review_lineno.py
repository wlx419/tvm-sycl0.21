#!/usr/bin/env python3
"""Generate a reviewer PDF source with blue additions and line numbers.

Deleted text is hidden so visible line locations remain close to the clean
revision. Added text is blue without underline or wave decoration.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


LINENO_PREAMBLE = r"""
%% Reviewer version: visible line numbers and blue revised text.
\usepackage[switch]{lineno}
\renewcommand\linenumberfont{\normalfont\scriptsize\color{gray}}
\setlength\linenumbersep{6pt}
\captionsetup{labelfont={color=blue},textfont={color=blue}}
\captionsetup[lstlisting]{labelfont={color=blue},textfont={color=blue}}
""".strip()


START_LINENO = r"""
% Reviewer version: deleted text is omitted; revised text is blue.
\linenumbers
""".strip()


def _listing_figure_bounds(text: str, label: str) -> tuple[int, int]:
    """Return the containing figure bounds for a labeled lstlisting."""
    marker = f"label={{{label}}}"
    label_pos = text.find(marker)
    if label_pos < 0:
        raise ValueError(f"Could not find listing label {label}")

    start = text.rfind(r"\begin{figure}", 0, label_pos)
    end_marker = r"\end{figure}"
    end = text.find(end_marker, label_pos)
    if start < 0 or end < 0:
        raise ValueError(f"Could not find figure containing {label}")
    return start, end + len(end_marker)


def _replace_listing_figures(diff_text: str, revised_text: str) -> str:
    """Use clean revised listings instead of latexdiff's commented code."""
    # latexdiff represents code-line additions/deletions as `%DIF >`/`%DIF <`.
    # Inside lstlisting those markers become visible code and invert the intended
    # reviewer view. All three listings changed, so show each clean revised
    # listing in blue as one reviewable unit.
    for label in ("lst:tir-to-sycl", "lst:local-mem", "lst:dispatch"):
        revised_start, revised_end = _listing_figure_bounds(revised_text, label)
        block = revised_text[revised_start:revised_end]
        block = re.sub(
            r"(\\begin\{figure\}(?:\[[^\]]*\])?)",
            r"\1\n\\color{blue}",
            block,
            count=1,
        )

        diff_start, diff_end = _listing_figure_bounds(diff_text, label)
        diff_text = diff_text[:diff_start] + block + diff_text[diff_end:]
    return diff_text


def _color_changed_blocks(text: str) -> str:
    """Color complete changed paragraphs or float blocks, including unchanged words."""
    marker = "%% main text"
    marker_pos = text.find(marker)
    if marker_pos < 0:
        raise ValueError("Could not find main-text marker")

    head = text[:marker_pos]
    body = text[marker_pos:]
    chunks = re.split(r"(\n[ \t]*\n)", body)
    environments = ("table", "table*", "figure", "figure*", "itemize", "enumerate", "description")

    for index in range(0, len(chunks), 2):
        block = chunks[index]
        if not any(token in block for token in (r"\DIFadd", r"\DIFaddFL")):
            continue
        if r"\color{blue}" in block and r"\DIFadd" not in block:
            continue

        begin_match = re.search(
            r"\\begin\{(" + "|".join(re.escape(env) for env in environments) + r")\}(?:\[[^\]]*\])?",
            block,
        )
        if begin_match:
            insert_at = begin_match.end()
            block = block[:insert_at] + "\n\\color{blue}" + block[insert_at:]
            env = begin_match.group(1)
            end_marker = rf"\end{{{env}}}"
            end_pos = block.rfind(end_marker)
            if end_pos >= 0:
                block = block[:end_pos] + "\\color{black}\n" + block[end_pos:]
            else:
                block += "\n\\color{black}"
        else:
            block = "\\color{blue}\n" + block + "\n\\color{black}"
        chunks[index] = block

    return head + "".join(chunks)


def _color_changed_captions(text: str) -> str:
    """Color a complete caption when any text within that caption changed."""
    cursor = 0
    while True:
        command = text.find(r"\caption", cursor)
        if command < 0:
            break
        opening = text.find("{", command + len(r"\caption"))
        if opening < 0:
            break
        depth = 0
        closing = -1
        for index in range(opening, len(text)):
            if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
                depth += 1
            elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    closing = index
                    break
        if closing < 0:
            raise ValueError("Unbalanced caption while coloring reviewer source")
        content = text[opening + 1 : closing]
        if any(token in content for token in (r"\DIFadd", r"\DIFaddFL")) and not content.lstrip().startswith(
            r"\color{blue}"
        ):
            text = text[: opening + 1] + r"\color{blue}" + content + text[closing:]
            closing += len(r"\color{blue}")
        cursor = closing + 1
    return text


def _color_captionof_text(text: str) -> str:
    """Color caption text emitted with captionof in newly added blocks."""
    return re.sub(
        r"(\\captionof\{(?:table|figure)\}\{)(?!\\color\{blue\})",
        r"\1\\color{blue}",
        text,
    )


def generate(original: Path, revised: Path, output: Path) -> None:
    result = subprocess.run(
        [
            "latexdiff",
            "--type=CFONT",
            "--no-del",
            "--graphics-markup=none",
            str(original),
            str(revised),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    text = result.stdout
    revised_text = revised.read_text(encoding="utf-8")
    text = _replace_listing_figures(text, revised_text)
    text = _color_changed_blocks(text)
    text = _color_changed_captions(text)
    text = _color_captionof_text(text)

    cite_line = "\\usepackage{cite}"
    if cite_line not in text:
        raise ValueError("Could not find preamble insertion point")
    text = text.replace(cite_line, cite_line + "\n\n" + LINENO_PREAMBLE, 1)

    # CFONT is robust for large rewritten paragraphs. Preserve the manuscript
    # font and use color alone rather than CFONT's sans-serif addition style.
    text = text.replace(
        r"\providecommand{\DIFadd}[1]{{\protect\color{blue} \sf #1}}",
        r"\providecommand{\DIFadd}[1]{{\protect\color{blue} #1}}",
    )
    text = text.replace(
        r"\providecommand{\DIFaddtex}[1]{{\protect\color{blue} \sf #1}}",
        r"\providecommand{\DIFaddtex}[1]{{\protect\color{blue} #1}}",
    )
    text = text.replace(r"[\color{blue}\sf]", r"[\color{blue}]")
    text = text.replace(r"[\color{blue}\sffamily]", r"[\color{blue}]")

    # latexdiff treats the complete twocolumn title/abstract wrapper as one
    # added block and does not place inline DIFadd commands inside it. Color
    # the revised abstract explicitly; leave structural begin/end markers as
    # no-ops because color commands there can break table alignment.
    text = text.replace(r"\begin{abstract}", r"\begin{abstract}\color{blue}", 1)
    text = text.replace(r"\end{abstract}", r"\color{black}\end{abstract}", 1)

    begin_document = "\\begin{document}"
    if begin_document not in text:
        raise ValueError("Could not find document insertion point")
    text = text.replace(
        begin_document,
        begin_document + "\n" + START_LINENO,
        1,
    )

    output.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--revised", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate(args.original, args.revised, args.out)


if __name__ == "__main__":
    main()
