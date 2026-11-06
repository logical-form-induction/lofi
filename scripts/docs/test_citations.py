"""Integration checks for the documentation's BibTeX citation forms."""

from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from markdown import Markdown
from mkdocs.exceptions import PluginError
from scripts.docs.citations import CitationExtension

ROOT = Path(__file__).resolve().parents[2]


class CitationTests(unittest.TestCase):
    """Check public citation behavior using the actual citation processor."""

    def render(self, source: str, bib_file: Path | None = None) -> ET.Element:
        """Render a fixture with the production citation extension."""
        md = Markdown(
            extensions=[
                "fenced_code",
                "footnotes",
                "pymdownx.arithmatex",
                CitationExtension(
                    bib_file or ROOT / "docs/references/references.bib",
                    ROOT / "docs/references/acl.csl",
                ),
            ]
        )
        html = md.convert(source)
        return ET.fromstring(f"<main>{html}</main>")  # noqa: S314 - Local test output.

    def test_four_forms_and_reference_placement(self) -> None:
        """Render each requested form with one linked reference entry."""
        root = self.render(r"""\citep{lambek1958}

\citet{lambek1958}

\citeauthor{lambek1958}'s (\citeyear{lambek1958})

\citealt{lambek1958}

## References

\bibliography
""")
        self.assertEqual(
            ["".join(node.itertext()) for node in root[:4]],
            [
                "(Lambek 1958)",
                "Lambek (1958)",
                "Lambek's (1958)",
                "Lambek 1958",
            ],
        )
        self.assertEqual(root[-1].get("id"), "refs")
        self.assertEqual(len(root[-1]), 1)
        self.assertIn(
            "The mathematics of sentence structure", "".join(root[-1].itertext())
        )
        self.assertEqual(root.findall(".//sup"), [])
        for link in root.findall(".//a"):
            target = link.attrib["href"]
            if target.startswith("#"):
                self.assertIsNotNone(root.find(f".//*[@id='{target[1:]}']"))

    def test_group_particles_and_code(self) -> None:
        """Preserve name particles and code while sorting the reference list."""
        root = self.render(r"""[@debruijn1972; @church1940]

`[@not-a-citation]` and `\citep{also-not-a-citation}` and `@property`.

```python
@unknown_decorator
text = "[@not-a-citation]"
```

## References

\bibliography
""")
        self.assertEqual("".join(root[0].itertext()), "(de Bruijn 1972; Church 1940)")
        refs = "".join(root[-1].itertext())
        self.assertIn("Church\N{EN DASH}Rosser", refs)
        self.assertIn("381\N{EN DASH}392", refs)
        self.assertEqual(len(root[-1]), 2)
        self.assertEqual(root[-1][0].get("id"), "ref-church1940")
        self.assertIn("[@not-a-citation]", "".join(root[1].itertext()))

    def test_unknown_key_and_missing_bibliography_fail(self) -> None:
        """Refuse unresolved citations and missing reference sections."""
        with self.assertRaises(PluginError):
            self.render("[@missing-lofi-key]\n\n## References\n\n\\bibliography")
        with self.assertRaises(PluginError):
            self.render("[@lambek1958]")

    def test_year_suffixes_consistent_across_forms(self) -> None:
        """Disambiguate same-author/year works consistently."""
        with tempfile.TemporaryDirectory() as directory:
            bib = Path(directory) / "fixture.bib"
            bib.write_text("""@book{a, author={Example, Ann}, year={2000},
title={Alpha}, publisher={Test}}
@book{b, author={Example, Ann}, year={2000}, title={Beta}, publisher={Test}}
""")
            root = self.render(
                r"""\citep{a,b}

\citet{a}; \citealt{b}; \citeauthor{b}'s (\citeyear{b}).

## References

\bibliography
""",
                bib,
            )
            self.assertEqual(
                "".join(root[0].itertext()), "(Example 2000a; Example 2000b)"
            )
            self.assertEqual(
                "".join(root[1].itertext()),
                "Example (2000a); Example 2000b; Example's (2000b).",
            )


if __name__ == "__main__":
    unittest.main()
