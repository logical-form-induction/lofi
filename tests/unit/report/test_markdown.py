from lofi.report.markdown import MARKDOWN, MarkdownRenderer


def test_a_markdown_table_writes_one_line_per_row_with_escaped_cells() -> None:
    rendered = MARKDOWN.render(
        ("Metric", "Value"),
        (("a|b", 0.5123), ("line\nbreak", None)),
        precision=2,
    ).decode()

    assert rendered.splitlines() == [
        "| Metric | Value |",
        "| --- | --- |",
        "| a\\|b | 0.51 |",
        "| line<br>break |  |",
    ]
    assert rendered.endswith("\n")


def test_a_markdown_table_is_written_with_the_markdown_suffix() -> None:
    assert MARKDOWN.suffix == ".md"
    assert repr(MarkdownRenderer()) == "MarkdownRenderer()"
