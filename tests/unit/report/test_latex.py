from lofi.report.latex import LATEX, LatexRenderer


def test_a_latex_table_writes_one_line_per_row_with_escaped_cells() -> None:
    rendered = LATEX.render(
        ("Metric_name", "Value"),
        (("50%", 0.5123), ("a&b", None)),
        precision=2,
    ).decode()

    assert rendered.splitlines() == [
        r"\begin{tabular}{ll}",
        r"Metric\_name & Value \\",
        r"\hline",
        r"50\% & 0.51 \\",
        r"a\&b &  \\",
        r"\end{tabular}",
    ]


def test_a_latex_table_is_written_with_the_tex_suffix() -> None:
    assert LATEX.suffix == ".tex"
    assert repr(LatexRenderer()) == "LatexRenderer()"
