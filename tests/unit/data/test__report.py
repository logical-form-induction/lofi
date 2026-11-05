import pytest

from lofi.data._reachability import (
    LabelBlocker,
    ReachabilityReport,
    ReachableDerivation,
)
from lofi.data._report import AnalysisCounts, LoadReport


def test_a_report_names_what_it_saw_emitted_and_set_aside() -> None:
    report = LoadReport(20, 17, {"no supervision arm": 2, "invalid record": 1})

    assert report.records_seen == 20
    assert report.emitted == 17
    assert report.skipped == 3
    assert report.skips_by_reason["no supervision arm"] == 2


def test_a_report_that_loses_records_between_its_counts_is_refused() -> None:
    with pytest.raises(ValueError, match="accounts for"):
        LoadReport(20, 17, {"invalid record": 1})


def test_a_report_of_a_load_that_skipped_nothing_needs_no_reasons() -> None:
    assert LoadReport(4, 4).skipped == 0


def test_a_skip_counted_under_no_reason_is_refused() -> None:
    with pytest.raises(ValueError, match="skip reason"):
        LoadReport(3, 2, {"": 1})


def test_a_negative_count_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        LoadReport(1, -1, {"invalid record": 2})


def test_a_rate_is_a_quotient_of_counts_rather_than_a_number_a_reader_states() -> None:
    report = LoadReport(10, 10, analyses=AnalysisCounts(8, 6, 5))

    assert report.reachability == 6 / 8
    assert report.normal_form_conformance == 5 / 8


def test_a_load_that_checked_no_analysis_has_no_rate_rather_than_a_zero() -> None:
    assert LoadReport(10, 10).reachability is None
    assert LoadReport(10, 10).normal_form_conformance is None
    assert AnalysisCounts(0).reachability is None
    assert AnalysisCounts(0).normal_form_conformance is None


def test_more_analyses_holding_than_were_checked_is_refused() -> None:
    with pytest.raises(ValueError, match="never looked at"):
        AnalysisCounts(3, 4)

    with pytest.raises(ValueError, match="never looked at"):
        AnalysisCounts(3, 0, 4)


def test_checking_an_analysis_of_an_item_that_was_not_emitted_is_refused() -> None:
    with pytest.raises(ValueError, match="did not emit"):
        LoadReport(10, 4, {"invalid record": 6}, AnalysisCounts(5))


def test_two_reports_of_one_load_are_the_same_value() -> None:
    one = LoadReport(5, 4, {"invalid record": 1}, AnalysisCounts(4, 3))
    other = LoadReport(5, 4, {"invalid record": 1}, AnalysisCounts(4, 3))

    assert one == other
    assert hash(one) == hash(other)
    assert one != LoadReport(5, 4, {"invalid record": 1}, AnalysisCounts(4, 2))
    assert one != LoadReport(5, 4, {"other reason": 1}, AnalysisCounts(4, 3))


def test_a_report_keeps_no_handle_on_the_mapping_it_was_given() -> None:
    reasons = {"invalid record": 1}
    report = LoadReport(3, 2, reasons)
    reasons["invalid record"] = 9

    assert report.skips_by_reason["invalid record"] == 1
    assert report.skipped == 1


def test_a_report_takes_on_nothing_after_it_is_built() -> None:
    report = LoadReport(1, 1)

    for field in ("records_seen", "emitted", "attached"):
        with pytest.raises(AttributeError):
            setattr(report, field, 2)


def test_a_report_reads_back_as_the_call_that_rebuilds_it() -> None:
    report = LoadReport(3, 2, {"invalid record": 1}, AnalysisCounts(2, 1, 1))

    assert repr(report) == (
        "LoadReport(3, 2, {'invalid record': 1}, AnalysisCounts(2, 1, 1))"
    )


def test_a_load_report_carries_matching_per_derivation_outcomes() -> None:
    reachability = ReachabilityReport(
        0,
        0,
        2,
        1,
        outcomes=(ReachableDerivation(), LabelBlocker(4, "missing")),
    )
    report = LoadReport(
        2,
        2,
        analyses=AnalysisCounts(2, 1, 1),
        reachability_report=reachability,
    )

    assert report.reachability_report is reachability
    assert report.reachability == 0.5
    assert hash(report) == hash(
        LoadReport(
            2,
            2,
            analyses=AnalysisCounts(2, 1, 1),
            reachability_report=reachability,
        )
    )
    assert "reachability_report=ReachabilityReport" in repr(report)


def test_a_load_report_rejects_inconsistent_reachability_counts() -> None:
    reachability = ReachabilityReport(
        0,
        0,
        1,
        1,
        outcomes=(ReachableDerivation(),),
    )

    with pytest.raises(ValueError, match="requires analysis counts"):
        LoadReport(1, 1, reachability_report=reachability)

    with pytest.raises(ValueError, match="does not match"):
        LoadReport(
            2,
            2,
            analyses=AnalysisCounts(2, 1),
            reachability_report=reachability,
        )

    with pytest.raises(ValueError, match="does not match"):
        LoadReport(
            1,
            1,
            analyses=AnalysisCounts(1, 0),
            reachability_report=reachability,
        )
