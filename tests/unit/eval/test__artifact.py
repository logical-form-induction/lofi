from typing import TYPE_CHECKING, Final, cast

import pytest

from lofi.eval._artifact import Hypothesis, HypothesisFeaturizer, LabelHead
from lofi.eval._inference import UndeclaredLabelError
from lofi.eval._ports import InferenceFeaturizer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_TWO_LABELS: Final = frozenset({"no", "yes"})


def _head(
    features: Sequence[Mapping[str, int]],
    labels: Sequence[str],
    inventory: frozenset[str] = _TWO_LABELS,
) -> LabelHead:
    return LabelHead(features, labels, inventory=inventory)


def _yes_no_head() -> LabelHead:
    return _head(
        [{"yes": 3}, {"yes": 2}, {"no": 3}, {"no": 2, "yes": 1}],
        ["yes", "yes", "no", "no"],
    )


# ---------------------------------------------------------------------------
# withholding the premise
# ---------------------------------------------------------------------------


def test_the_hypothesis_featurizer_names_no_premise_token() -> None:
    features = HypothesisFeaturizer()(Hypothesis(("the", "dog", "the")))

    assert dict(features) == {"hypothesis:the": 2, "hypothesis:dog": 1}
    assert not any(key.startswith("premise:") for key in features)


def test_the_hypothesis_featurizer_refuses_a_pair_that_carries_a_premise() -> None:
    pair = (("a", "premise"), ("a", "hypothesis"))

    with pytest.raises(TypeError, match="takes no premise"):
        HypothesisFeaturizer()(cast("Hypothesis", pair))


def test_the_hypothesis_featurizer_refuses_an_empty_token() -> None:
    with pytest.raises(ValueError, match="hypothesis token"):
        HypothesisFeaturizer()(Hypothesis(("the", "")))


def test_the_hypothesis_featurizer_satisfies_the_featurizer_protocol() -> None:
    assert isinstance(HypothesisFeaturizer(), InferenceFeaturizer)
    assert HypothesisFeaturizer().name == "hypothesis-tokens"
    assert repr(HypothesisFeaturizer()) == "HypothesisFeaturizer()"


# ---------------------------------------------------------------------------
# the head
# ---------------------------------------------------------------------------


def test_a_head_reads_the_label_its_features_point_to() -> None:
    head = _yes_no_head()

    assert head.predict({"yes": 4}) == "yes"
    assert head.predict({"no": 4}) == "no"


def test_a_head_scores_every_declared_label_in_ascending_order() -> None:
    scored = _yes_no_head().scores({"yes": 1})

    assert list(scored) == ["no", "yes"]
    assert scored["yes"] > scored["no"]


def test_a_head_records_what_it_was_fitted_on() -> None:
    head = _yes_no_head()

    assert head.fitted == 4
    assert head.inventory == _TWO_LABELS
    assert head.vocabulary == frozenset({"yes", "no"})


def test_a_head_ignores_a_feature_the_fitted_split_never_carried() -> None:
    head = _yes_no_head()

    assert head.scores({"yes": 1}) == head.scores({"yes": 1, "unseen": 9})


def test_a_head_breaks_a_score_tie_toward_the_fitted_majority() -> None:
    head = _head(
        [{"shared": 1}, {"shared": 1}, {"shared": 1}],
        ["yes", "yes", "no"],
    )

    assert head.majority == "yes"
    assert head.predict({"shared": 1}) == "yes"


def test_a_fitted_label_head_cannot_be_mutated() -> None:
    head = LabelHead(({}, {}), ("no", "yes"), inventory=frozenset({"no", "yes"}))
    with pytest.raises(AttributeError, match="immutable"):
        head.__setattr__("fitted", 100)


def test_a_head_breaks_a_majority_tie_toward_the_first_label_ascending() -> None:
    head = _head([{"shared": 1}, {"shared": 1}], ["yes", "no"])

    assert head.majority == "no"
    assert head.predict({"shared": 1}) == "no"


def test_a_head_refuses_fewer_than_two_declared_labels() -> None:
    with pytest.raises(ValueError, match="classifies nothing"):
        _head([{"a": 1}], ["yes"], frozenset({"yes"}))


def test_a_head_refuses_features_and_labels_of_different_lengths() -> None:
    with pytest.raises(ValueError, match="2 featurized items and 1 labels"):
        _head([{"yes": 1}, {"no": 1}], ["yes"])


def test_a_head_refuses_an_empty_fitted_split() -> None:
    with pytest.raises(ValueError, match="fitted on no item"):
        _head([], [])


def test_a_head_refuses_a_fitted_label_outside_the_inventory() -> None:
    with pytest.raises(UndeclaredLabelError, match="not in the declared inventory"):
        _head([{"yes": 1}, {"no": 1}], ["yes", "maybe"])


def test_a_head_refuses_a_declared_label_the_fitted_split_never_shows() -> None:
    with pytest.raises(ValueError, match="no item labeled"):
        _head([{"yes": 1}, {"yes": 1}], ["yes", "yes"])


def test_a_head_refuses_a_negative_feature_count() -> None:
    with pytest.raises(ValueError, match="count of feature"):
        _head([{"yes": -1}, {"no": 1}], ["yes", "no"])

    with pytest.raises(ValueError, match="count of feature"):
        _yes_no_head().predict({"yes": -1})


def test_a_head_reprs_its_inventory_its_fitted_count_and_its_vocabulary() -> None:
    assert repr(_yes_no_head()) == (
        "LabelHead(inventory=['no', 'yes'], fitted=4, vocabulary=2)"
    )


# ---------------------------------------------------------------------------
# the arm and the floor as values
# ---------------------------------------------------------------------------
