"""Declaration tests for the globally normalized span parser."""

import didactic.api as dx
import pytest

from lofi.spec import ParserSpec
from lofi.spec._ports import Dependency
from lofi.spec.parsers.span_crf import SpanCrfParserSpec


def test_the_span_crf_is_one_registered_chart_variant() -> None:
    declared = ParserSpec.model_validate({"kind": "span-crf", "hidden": 7})

    assert type(declared) is SpanCrfParserSpec
    assert declared == SpanCrfParserSpec(hidden=7)
    assert ParserSpec.__variants__["span-crf"] is SpanCrfParserSpec


def test_the_span_crf_reads_only_pooled_words() -> None:
    declared = SpanCrfParserSpec(hidden=7)

    assert declared.dependencies() == (Dependency("subword_pooler", "leaves"),)
    assert declared.chart_outputs() == {
        "inside",
        "outside",
        "alpha",
        "mu",
        "log_z",
        "entropy",
        "mbr",
    }


def test_the_span_crf_hidden_width_is_positive() -> None:
    with pytest.raises(dx.ValidationError, match="hidden"):
        SpanCrfParserSpec(hidden=0)
