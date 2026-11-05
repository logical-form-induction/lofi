from importlib.metadata import entry_points

import pytest

from lofi.eval import RESIDUAL_SOURCES
from lofi.report import (
    DERIVED_COLUMNS,
    REPORT_FIELDS,
    REPORT_FORMAT_FAMILY,
    ReportRenderer,
    format_cell,
    report_columns,
    report_renderer,
    report_stem,
)
from lofi.report.latex import LATEX
from lofi.report.markdown import MARKDOWN


def test_the_schema_holds_one_residual_column_per_residual_source() -> None:
    names = [field.name for field in REPORT_FIELDS]
    residuals = [name for name in names if name.startswith("residual_")]

    assert residuals == [
        "residual_measured",
        *(f"residual_{source}" for source in RESIDUAL_SOURCES),
    ]
    assert names[:3] == [
        "freeze_id",
        "config_hash",
        "grammar_fingerprint",
    ]


def test_a_report_shows_stored_and_derived_columns_and_nothing_else() -> None:
    declared = (*(field.name for field in REPORT_FIELDS), *DERIVED_COLUMNS)

    assert report_columns(declared) == declared

    with pytest.raises(ValueError, match="unknown columns"):
        report_columns(("metric", "not-a-column"))


@pytest.mark.parametrize(
    "name", ["../headline", "tables/headline", r"tables\headline", "..", " "]
)
def test_a_report_name_cannot_leave_the_results_directory(name: str) -> None:
    with pytest.raises(ValueError, match="filename stem"):
        report_stem(name)


def test_a_report_name_that_is_a_stem_is_kept() -> None:
    assert report_stem("headline") == "headline"


def test_every_installed_report_format_names_a_renderer() -> None:
    installed = {
        point.name: point.load()
        for point in entry_points(group=f"lofi.{REPORT_FORMAT_FAMILY}")
    }

    assert installed == {"markdown": MARKDOWN, "latex": LATEX}
    assert all(isinstance(member, ReportRenderer) for member in installed.values())
    assert report_renderer("markdown") is MARKDOWN
    assert report_renderer("latex") is LATEX


def test_an_uninstalled_report_format_is_refused() -> None:
    with pytest.raises(ValueError, match="report_format"):
        report_renderer("html")


def test_a_cell_is_written_at_the_declared_precision() -> None:
    assert format_cell(None, 2) == ""
    assert format_cell(0.125, 2) == "0.12"
    assert format_cell(0.125, None) == "0.125"
    assert format_cell(3, 2) == "3"
    assert format_cell("recovery", 2) == "recovery"

    with pytest.raises(TypeError, match="not scalar"):
        format_cell([1.0], 2)
