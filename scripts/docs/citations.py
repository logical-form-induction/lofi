"""Render BibTeX citations through Pandoc citeproc in Python-Markdown.

MkDocs loads this module as a hook. Only citation nodes go through Pandoc;
MkDocs retains responsibility for code, mathematics, admonitions, and links.
"""

from __future__ import annotations

import copy
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

if TYPE_CHECKING:
    from markdown import Markdown
    from mkdocs.config.defaults import MkDocsConfig

import pypandoc
from markdown.extensions import Extension
from markdown.inlinepatterns import InlineProcessor
from markdown.treeprocessors import Treeprocessor
from markdown.util import AtomicString
from mkdocs.exceptions import PluginError

KEY = r"[\w:.-]+"
COMMAND = rf"\\(citep|citet|citealt|citeauthor|citeyear)\{{({KEY}(?:\s*,\s*{KEY})*)\}}"
BRACKET = rf"\[(@{KEY}(?:\s*;\s*@{KEY})*)\]"


class CitationPattern(InlineProcessor):
    """Collect citations after code spans have been protected by Markdown."""

    ANCESTOR_EXCLUDES = ("code", "pre", "a")

    @override
    def handleMatch(
        self,
        m: re.Match[str],
        data: str,
    ) -> tuple[ET.Element, int, int]:
        """Return a placeholder for the matched citation command."""
        del data
        node = ET.Element("lofi-citation")
        if m.group(1):
            node.set("mode", m.group(1))
            node.set("keys", re.sub(r"\s+", "", m.group(2)))
        else:
            node.set("mode", "citep")
            node.set("keys", ",".join(re.findall(rf"@({KEY})", m.group(3))))
        return node, m.start(0), m.end(0)


class BibliographyPattern(InlineProcessor):
    """Mark the position of the generated reference list."""

    @override
    def handleMatch(
        self,
        m: re.Match[str],
        data: str,
    ) -> tuple[ET.Element, int, int]:
        """Return a placeholder for the matched citation command."""
        del data
        return ET.Element("lofi-bibliography"), m.start(0), m.end(0)


def render_citations(
    requests: list[tuple[str, list[str]]],
    bib_file: Path,
    csl_file: Path,
) -> tuple[list[ET.Element], ET.Element]:
    """Resolve all citations together, including same-author/year suffixes."""
    source = []
    for mode, keys in requests:
        if mode != "citep" and len(keys) != 1:
            message = f"{mode} requires one citation key; use citep for a group"
            raise PluginError(message)
        source.append(
            "[" + "; ".join(f"@{key}" for key in keys) + "]"
            if mode == "citep"
            else f"@{keys[0]}"
        )
    document = json.loads(
        pypandoc.convert_text(
            "\n\n".join(source),
            "json",
            format="markdown",
            extra_args=[
                "--citeproc",
                "--fail-if-warnings",
                f"--bibliography={bib_file}",
                f"--csl={csl_file}",
                "-M",
                "link-citations=true",
            ],
        )
    )
    blocks = document["blocks"]
    if len(blocks) != len(requests) + 1 or blocks[-1]["t"] != "Div":
        message = "Unexpected Pandoc bibliography structure"
        raise PluginError(message)
    for block, (mode, _) in zip(blocks[:-1], requests, strict=True):
        if mode in {"citep", "citet"}:
            continue
        # Pandoc's AuthorInText result ends with Space, '(', linked year, ')'.
        # Work on structured nodes, so author particles and year suffixes survive.
        parts = block["c"][0]["c"][1]
        if (
            len(parts) < 5
            or parts[-4]["t"] != "Space"
            or parts[-3] != {"t": "Str", "c": "("}
            or parts[-2]["t"] != "Link"
            or parts[-1] != {"t": "Str", "c": ")"}
        ):
            message = "Unexpected Pandoc narrative citation structure"
            raise PluginError(message)
        author, year = parts[:-4], parts[-2]
        replacement = {
            "citealt": [*author, {"t": "Space"}, year],
            "citeauthor": author,
            "citeyear": [year],
        }[mode]
        block["c"][0]["c"][1] = replacement
    html = pypandoc.convert_text(
        json.dumps(document), "html5", format="json", extra_args=["--wrap=none"]
    )
    root = ET.fromstring(f"<div>{html}</div>")
    rendered = list(root)
    # Prevent later Markdown processors from interpreting generated punctuation.
    for node in root.iter():
        if node.text:
            node.text = AtomicString(node.text)
        if node.tail:
            node.tail = AtomicString(node.tail)
    return [node[0] for node in rendered[:-1]], rendered[-1]


class CitationTreeprocessor(Treeprocessor):
    """Replace citation placeholders with citeproc output for this page."""

    def __init__(self, md: Markdown, bib_file: Path, csl_file: Path) -> None:
        super().__init__(md)
        self.bib_file = bib_file
        self.csl_file = csl_file

    @override
    def run(self, root: ET.Element) -> None:
        """Resolve the page citations and append its reference list."""
        citations = list(root.iter("lofi-citation"))
        markers = list(root.iter("lofi-bibliography"))
        if not citations and not markers:
            return
        if len(markers) != 1:
            message = "A cited page must end with References and one \\bibliography"
            raise PluginError(message)
        parents = {child: parent for parent in root.iter() for child in parent}
        marker_paragraph = parents[markers[0]]
        if (
            marker_paragraph.tag != "p"
            or len(marker_paragraph) != 1
            or (marker_paragraph.text or "").strip()
            or (markers[0].tail or "").strip()
            or list(root)[-1] is not marker_paragraph
            or len(root) < 2
            or root[-2].tag != "h2"
            or "".join(root[-2].itertext()).strip() != "References"
        ):
            message = (
                "Put \\bibliography alone after ## References at the end of the page"
            )
            raise PluginError(message)
        if not citations:
            root.remove(marker_paragraph)
            return
        requests = [
            (node.attrib["mode"], node.attrib["keys"].split(",")) for node in citations
        ]
        try:
            formatted, bibliography = render_citations(
                requests, self.bib_file, self.csl_file
            )
        except (RuntimeError, ValueError, ET.ParseError) as error:
            message = f"Could not render BibTeX citations: {error}"
            raise PluginError(message) from error
        for original, rendered in zip(citations, formatted, strict=True):
            replacement = copy.deepcopy(rendered)
            replacement.tail = original.tail
            parent = parents[original]
            index = list(parent).index(original)
            parent.remove(original)
            parent.insert(index, replacement)
        root.remove(marker_paragraph)
        root.append(bibliography)


class CitationExtension(Extension):
    """Add BibTeX citation commands to Python-Markdown."""

    def __init__(self, bib_file: Path, csl_file: Path) -> None:
        self.bib_file, self.csl_file = bib_file, csl_file
        super().__init__()

    @override
    def extendMarkdown(self, md: Markdown) -> None:
        """Register citation parsing and bibliography rendering."""
        # Below backticks (190), above backslash escapes (180).
        md.inlinePatterns.register(
            CitationPattern(f"(?:{COMMAND})|(?:{BRACKET})", md), "lofi-citations", 185
        )
        md.inlinePatterns.register(
            BibliographyPattern(r"\\bibliography\b", md), "lofi-bibliography", 185
        )
        md.treeprocessors.register(
            CitationTreeprocessor(md, self.bib_file, self.csl_file), "lofi-citeproc", 15
        )


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    """Install the citation extension in the MkDocs Markdown configuration."""
    root = Path(config.config_file_path).resolve().parent
    # Python-Markdown accepts Extension instances; MkDocs annotates strings only.
    extensions = cast("list[str | Extension]", config.markdown_extensions)
    extensions.append(
        CitationExtension(
            root / "docs/references/references.bib",
            root / "docs/references/acl.csl",
        )
    )
    return config
