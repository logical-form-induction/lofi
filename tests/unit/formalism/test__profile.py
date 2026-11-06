from copy import copy, deepcopy
from typing import cast

import pytest

from lofi.formalism._profile import (
    BoundedConsumer,
    CategorySet,
    DeductionProfile,
    ProfileBound,
    ProfileUnsupportedError,
)


def _span_bound() -> ProfileBound:
    return ProfileBound("DeepInsideOutside", 1, frozenset({1, 2}), "RegionIndex")


def _symbolic_bound() -> ProfileBound:
    return ProfileBound("lofi.deduce.parse", 1, frozenset({1, 2, 3}), "RegionIndex")


def _tagged_bound() -> ProfileBound:
    return ProfileBound("TaggedInsideChart", 1, frozenset({1, 2}), "RegionIndex")


def _profile(
    fanout: int,
    ranks: frozenset[int],
    *,
    category_set: CategorySet = "unknown",
) -> DeductionProfile:
    return DeductionProfile(
        fanout,
        ranks,
        unary_closure=1 in ranks,
        category_set=category_set,
        lexicalized=True,
    )


def test_a_profile_reports_what_the_formalism_demands() -> None:
    profile = _profile(1, frozenset({1, 2}), category_set="infinite")

    assert profile.fanout == 1
    assert profile.ranks == frozenset({1, 2})
    assert profile.unary_closure
    assert profile.category_set == "infinite"
    assert profile.lexicalized


def test_profiles_are_equal_on_all_five_fields() -> None:
    profile = _profile(1, frozenset({1, 2}))

    assert profile == _profile(1, frozenset({1, 2}))
    assert hash(profile) == hash(_profile(1, frozenset({1, 2})))
    assert profile != _profile(2, frozenset({1, 2}))
    assert profile != _profile(1, frozenset({2}))
    assert profile != _profile(1, frozenset({1, 2}), category_set="finite")
    assert profile != (1, frozenset({1, 2}))


def test_a_profile_reprs_as_the_call_that_rebuilds_it() -> None:
    text = repr(_profile(1, frozenset({2})))

    assert text.startswith("DeductionProfile(1, frozenset({2})")
    assert text.endswith("category_set='unknown', lexicalized=True)")


@pytest.mark.parametrize("fanout", [0, -1])
def test_a_profile_below_fan_out_one_is_refused(fanout: int) -> None:
    with pytest.raises(ValueError, match="fan-out"):
        _profile(fanout, frozenset({2}))


def test_a_profile_and_bound_accept_rank_zero() -> None:
    profile = _profile(1, frozenset({0, 2}))
    bound = ProfileBound("axiom chart", 1, frozenset({0, 2}))

    assert profile.ranks == frozenset({0, 2})
    assert bound.ranks == frozenset({0, 2})
    bound.check(profile, "axiomatic")


def test_a_negative_arity_is_refused() -> None:
    with pytest.raises(ValueError, match="arity"):
        _profile(1, frozenset({-1, 2}))

    with pytest.raises(ValueError, match="arity"):
        ProfileBound("a chart", 1, frozenset({-1, 2}))


def test_a_fan_out_that_is_a_flag_rather_than_a_count_is_refused() -> None:
    fanout: bool = True

    with pytest.raises(TypeError, match="fan-out"):
        _profile(fanout, frozenset({2}))


def test_a_unary_closure_over_no_unary_rule_is_refused() -> None:
    with pytest.raises(ValueError, match="unary"):
        DeductionProfile(
            1,
            frozenset({2}),
            unary_closure=True,
            category_set="finite",
            lexicalized=True,
        )


def test_the_span_chart_serves_a_unary_and_binary_formalism() -> None:
    _span_bound().check(_profile(1, frozenset({1, 2})), "ccg")


def test_the_span_chart_refuses_fan_out_two_by_naming_its_extension_point() -> None:
    with pytest.raises(ProfileUnsupportedError) as refusal:
        _span_bound().check(_profile(2, frozenset({1, 2})), "nl-diamond")

    message = str(refusal.value)
    extension = _span_bound().extension_point or ""

    assert extension
    assert _span_bound().consumer in message
    assert "nl-diamond" in message
    assert "fan-out 2" in message
    assert extension in message


def test_the_symbolic_and_neural_bounds_disagree_about_ternary_coordination() -> None:
    profile = _profile(1, frozenset({1, 2, 3}))

    _symbolic_bound().check(profile, "ccg")

    with pytest.raises(ProfileUnsupportedError) as refusal:
        _span_bound().check(profile, "ccg")

    with pytest.raises(ProfileUnsupportedError, match=r"\[3\]"):
        _tagged_bound().check(profile, "ccg")

    message = str(refusal.value)

    assert "[3]" in message
    assert "[1, 2]" in message


def test_the_consumer_bounds_agree_about_fan_out() -> None:
    profile = _profile(2, frozenset({2}))

    for bound in (_span_bound(), _symbolic_bound(), _tagged_bound()):
        with pytest.raises(ProfileUnsupportedError, match="fan-out"):
            bound.check(profile, "tag")


def test_a_consumer_checks_rule_arities_without_a_forged_profile() -> None:
    _tagged_bound().check_ranks(frozenset({1, 2}), "ccg")

    with pytest.raises(ProfileUnsupportedError) as refusal:
        _tagged_bound().check_ranks(frozenset({1, 2, 3}), "coordination")

    message = str(refusal.value)

    assert _tagged_bound().consumer in message
    assert "coordination" in message
    assert "[3]" in message


def test_a_bound_with_no_extension_point_refuses_without_naming_one() -> None:
    bound = ProfileBound("a chart", 1, frozenset({2}))

    with pytest.raises(ProfileUnsupportedError) as refusal:
        bound.check(_profile(2, frozenset({2})), "tag")

    assert "Extension point" not in str(refusal.value)


def test_a_bound_reports_what_it_serves_and_what_would_raise_it() -> None:
    bound = ProfileBound("a chart", 1, frozenset({1, 2}), "a region index")

    assert bound.consumer == "a chart"
    assert bound.max_fanout == 1
    assert bound.ranks == frozenset({1, 2})
    assert bound.extension_point == "a region index"
    assert bound == ProfileBound("a chart", 1, frozenset({1, 2}), "a region index")
    assert hash(bound) == hash(
        ProfileBound("a chart", 1, frozenset({1, 2}), "a region index")
    )
    assert bound != ProfileBound("a chart", 1, frozenset({1, 2}))
    assert bound != _span_bound()
    assert repr(bound).startswith("ProfileBound('a chart', 1, ")


@pytest.mark.parametrize(
    ("consumer", "max_fanout", "extension_point"),
    [
        ("", 1, None),
        ("a chart", 0, None),
        ("a chart", 1, ""),
    ],
)
def test_an_ill_formed_bound_is_refused(
    consumer: str, max_fanout: int, extension_point: str | None
) -> None:
    with pytest.raises(ValueError, match="bound"):
        ProfileBound(consumer, max_fanout, frozenset({2}), extension_point)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("unary_closure", cast("bool", 1), TypeError),
        ("category_set", "bounded", ValueError),
        ("lexicalized", cast("bool", 1), TypeError),
    ],
)
def test_an_invalid_profile_field_is_refused(
    field: str,
    value: int | str,
    error: type[Exception],
) -> None:
    arguments = {
        "unary_closure": False,
        "category_set": "finite",
        "lexicalized": True,
    }
    arguments[field] = value

    with pytest.raises(error):
        DeductionProfile(1, frozenset({2}), **arguments)


def test_profiles_and_bounds_are_frozen() -> None:
    profile = _profile(1, frozenset({2}))
    bound = ProfileBound("consumer", 1, frozenset({2}))

    assert copy(profile) is profile
    assert deepcopy(profile) is profile
    assert copy(bound) is bound
    assert deepcopy(bound) is bound

    with pytest.raises(AttributeError, match="immutable"):
        profile.__setattr__("_fanout", 2)

    with pytest.raises(AttributeError, match="immutable"):
        bound.__setattr__("_max_fanout", 2)


class _Chart:
    def profile_bound(self) -> ProfileBound:
        return _span_bound()


def test_a_declaration_publishing_its_bound_is_a_bounded_consumer() -> None:
    assert isinstance(_Chart(), BoundedConsumer)
    assert not isinstance(object(), BoundedConsumer)
