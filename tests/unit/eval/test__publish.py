import socket
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, NamedTuple, cast

import pytest
from lairs.records import annotation as annotation_records

import lofi.eval
from lofi._types import ItemId
from lofi.data import LicenseRef, Licensing, Provenance, Reproducibility
from lofi.eval._matched import InducedBracketing
from lofi.eval._publish import (
    LAYERS_TARGET,
    LayersPredictionLens,
    PredictedAnalysis,
    PublishLens,
    PublishTarget,
)

if TYPE_CHECKING:
    from collections.abc import Callable

ITEM_ID = ItemId("at://did:plc:lofi/pub.layers.expression.expression/3k2")
TOKENIZATION = "1f0a6d1e-0d3c-4a1b-9f77-0c4b9c2f1d55"
EPRINT = "at://did:plc:lofi/pub.layers.eprint.eprint/1"
PRODUCED_AT = datetime(2026, 9, 22, 14, 30, tzinfo=UTC)
INDUCED = "automatic"


def test_the_generic_evaluation_surface_exposes_no_hub_route() -> None:
    for name in ("HUB_TARGET", "HubCardLens", "ModelRelease"):
        assert name not in lofi.eval.__all__
        assert not hasattr(lofi.eval, name)


def test_the_installed_layers_publisher_renders_a_prediction() -> None:
    registered = {entry.name: entry for entry in entry_points(group="lofi.publisher")}
    publisher = registered["layers-records"].load()("lofi", "ccg")

    layer = annotation_records.AnnotationLayer.model_validate_json(
        publisher.render(_analysis())
    )

    assert layer.expression == ITEM_ID
    assert layer.sourceMethod == INDUCED


class _NetworkUsedError(Exception):
    """Raised where a publish lens opens a socket."""


def _refusing[**P, R](_replaced: Callable[P, R]) -> Callable[P, R]:
    def refuse(*_args: P.args, **_kwargs: P.kwargs) -> R:
        raise _NetworkUsedError

    return refuse


def _provenance(source_method: str = "converted-corrected") -> Provenance:
    return Provenance(
        source_method,
        Licensing(
            "CC-BY-4.0 AND LicenseRef-LDC-User-Agreement",
            (
                LicenseRef("CC-BY-4.0", "annotations"),
                LicenseRef("LicenseRef-LDC-User-Agreement", "underlying-text"),
            ),
        ),
        (EPRINT,),
        Reproducibility(
            "https://github.com/logical-form-induction/lofi",
            "0f1e2d3c4b5a",
            "lofi run parse",
            "uv-3.14",
            17,
        ),
    )


def _bracketing() -> InducedBracketing:
    return InducedBracketing(
        4, {(0, 4): "S", (0, 1): "NP", (1, 4): "S\\NP", (2, 4): "NP"}
    )


def _analysis(
    source_method: str = "converted-corrected", *, item_id: ItemId = ITEM_ID
) -> PredictedAnalysis:
    return PredictedAnalysis(
        item_id,
        TOKENIZATION,
        _bracketing(),
        _provenance(source_method),
        PRODUCED_AT,
    )


def _layer_terms(text: str) -> tuple[str, str]:
    layer = annotation_records.AnnotationLayer.model_validate_json(text)
    licensing = layer.licensing
    expression = "" if licensing is None else licensing.expression or ""

    return expression, layer.sourceMethod or ""


class PublishLensFixture[S](NamedTuple):
    """A lens, what it publishes, and how the artifact is read back.

    Attributes
    ----------
    lens : PublishLens[S]
        The lens under test.
    source : S
        The result the lens renders.
    twin : S
        A separately built value equal to ``source``, so a test can tell a
        mutated source from an unmutated one.
    republished : Callable[[str], tuple[str, str]]
        Reads the license expression and source method back out of rendered
        artifact text, through the target's own validation.
    provenance : Provenance
        The terms the source inherited from its corpus.
    source_method : str
        The method the published artifact declares.
    """

    lens: PublishLens[S]
    source: S
    twin: S
    republished: Callable[[str], tuple[str, str]]
    provenance: Provenance
    source_method: str

    def render(self) -> str:
        return self.lens.render(self.source)


type _AnyFixture = PublishLensFixture[PredictedAnalysis]


class PublishLensLaws:
    pytestmark = pytest.mark.conformance

    def test_the_target_names_itself_and_the_media_type_it_takes(
        self, publish_lens: _AnyFixture
    ) -> None:
        target = publish_lens.lens.target

        assert isinstance(target, PublishTarget)
        assert target.tag
        assert target.media_type

    def test_the_artifact_carries_the_corpus_terms_forward(
        self, publish_lens: _AnyFixture
    ) -> None:
        expression, source_method = publish_lens.republished(publish_lens.render())

        assert expression == publish_lens.provenance.licensing.expression
        assert source_method == publish_lens.source_method
        assert expression
        assert source_method

    def test_rendering_the_same_result_twice_gives_the_same_artifact(
        self, publish_lens: _AnyFixture
    ) -> None:
        assert publish_lens.render() == publish_lens.render()

    def test_rendering_does_not_change_what_it_published(
        self, publish_lens: _AnyFixture
    ) -> None:
        publish_lens.render()

        assert publish_lens.source == publish_lens.twin

    def test_rendering_opens_no_socket(
        self, publish_lens: _AnyFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(socket, "socket", _refusing(socket.socket))
        monkeypatch.setattr(
            socket, "create_connection", _refusing(socket.create_connection)
        )

        assert publish_lens.render()


class TestLayersPredictionLens(PublishLensLaws):
    @pytest.fixture
    def publish_lens(self) -> PublishLensFixture[PredictedAnalysis]:
        return PublishLensFixture(
            LayersPredictionLens("lofi", "ccg"),
            _analysis(),
            _analysis(),
            _layer_terms,
            _provenance(),
            INDUCED,
        )

    def test_the_lens_renders_for_the_layers_repository(self) -> None:
        assert LayersPredictionLens("lofi", "ccg").target == LAYERS_TARGET
        assert LAYERS_TARGET.media_type == "application/json"

    def test_a_published_record_round_trips_through_the_layers_models(self) -> None:
        text = LayersPredictionLens("lofi", "ccg").render(_analysis())
        layer = annotation_records.AnnotationLayer.model_validate_json(text)

        assert _analysis().item_id == ITEM_ID
        assert layer.expression == ITEM_ID
        assert layer.formalism == "ccg"
        assert layer.kind == "tree"
        assert layer.subkind == "constituency"
        assert layer.tokenizationId is not None
        assert layer.tokenizationId.value == TOKENIZATION
        assert layer.model_dump_json() == text

    def test_the_component_licenses_survive_the_round_trip(self) -> None:
        text = LayersPredictionLens("lofi", "ccg").render(_analysis())
        layer = annotation_records.AnnotationLayer.model_validate_json(text)

        assert layer.licensing is not None
        assert tuple(
            (reference.spdx, reference.appliesTo)
            for reference in layer.licensing.licenses
        ) == (
            ("CC-BY-4.0", "annotations"),
            ("LicenseRef-LDC-User-Agreement", "underlying-text"),
        )

    def test_the_corpus_reproducibility_record_survives_the_round_trip(self) -> None:
        text = LayersPredictionLens("lofi", "ccg").render(_analysis())
        layer = annotation_records.AnnotationLayer.model_validate_json(text)

        assert layer.eprintRefs == (EPRINT,)
        assert layer.reproducibility is not None
        assert layer.reproducibility.randomSeed == 17
        assert layer.reproducibility.commitHash == "0f1e2d3c4b5a"
        assert layer.reproducibility.codeUri == (
            "https://github.com/logical-form-induction/lofi"
        )

    def test_every_predicted_span_becomes_one_anchored_annotation(self) -> None:
        text = LayersPredictionLens("lofi", "ccg").render(_analysis())
        layer = annotation_records.AnnotationLayer.model_validate_json(text)
        anchored = {}

        for annotation in layer.annotations:
            assert annotation.anchor is not None
            assert annotation.anchor.tokenRefSequence is not None
            sequence = annotation.anchor.tokenRefSequence

            assert sequence.tokenizationId.value == TOKENIZATION

            anchored[sequence.tokenIndexes] = annotation.label

        assert anchored == {
            (0, 1, 2, 3): "S",
            (0,): "NP",
            (1, 2, 3): "S\\NP",
            (2, 3): "NP",
        }

    def test_a_constituent_is_the_child_of_the_shortest_span_containing_it(
        self,
    ) -> None:
        text = LayersPredictionLens("lofi", "ccg").render(_analysis())
        layer = annotation_records.AnnotationLayer.model_validate_json(text)
        parents = {
            annotation.uuid.value: (
                None if annotation.parentId is None else annotation.parentId.value
            )
            for annotation in layer.annotations
        }
        children = {
            annotation.uuid.value: tuple(
                child.value for child in annotation.childIds or ()
            )
            for annotation in layer.annotations
        }

        assert parents == {"0-4": None, "0-1": "0-4", "1-4": "0-4", "2-4": "1-4"}
        assert children["0-4"] == ("0-1", "1-4")
        assert children["1-4"] == ("2-4",)
        assert children["2-4"] == ()

    def test_the_annotation_metadata_names_the_producing_tool(self) -> None:
        text = LayersPredictionLens("supertagger", "ccg").render(_analysis())
        layer = annotation_records.AnnotationLayer.model_validate_json(text)

        assert layer.metadata is not None
        assert layer.metadata.tool == "supertagger"
        assert layer.createdAt == PRODUCED_AT

    def test_a_prediction_is_marked_as_induced_not_as_its_source_corpus(self) -> None:
        text = LayersPredictionLens("lofi", "ccg").render(_analysis("crowd-sourced"))
        layer = annotation_records.AnnotationLayer.model_validate_json(text)

        assert layer.sourceMethod == INDUCED

    def test_three_items_with_one_surface_round_trip_as_three_records(self) -> None:
        fixture = tuple(
            (
                ItemId(
                    "at://did:plc:lofi/pub.layers.expression.expression/"
                    f"same-surface-{index}"
                ),
                "the cat sleeps",
            )
            for index in range(3)
        )
        lens = LayersPredictionLens("lofi", "ccg")
        layers = tuple(
            annotation_records.AnnotationLayer.model_validate_json(
                lens.render(_analysis(item_id=item_id))
            )
            for item_id, _surface in fixture
        )

        assert len({surface for _, surface in fixture}) == 1
        assert tuple(ItemId(layer.expression) for layer in layers) == tuple(
            item_id for item_id, _ in fixture
        )
        assert len({layer.expression for layer in layers}) == 3
        assert all(layer.sourceMethod == INDUCED for layer in layers)
        assert all(
            layer.licensing is not None and bool(layer.licensing.expression)
            for layer in layers
        )

    def test_an_analysis_without_an_item_id_is_not_constructible(self) -> None:
        with pytest.raises(ValueError, match="analysis's item id"):
            PredictedAnalysis(
                cast("ItemId", ""),
                TOKENIZATION,
                _bracketing(),
                _provenance(),
                PRODUCED_AT,
            )

    def test_an_analysis_without_a_tokenization_is_not_constructible(self) -> None:
        with pytest.raises(ValueError, match="analysis's tokenization"):
            PredictedAnalysis(ITEM_ID, "", _bracketing(), _provenance(), PRODUCED_AT)

    def test_a_lens_without_a_formalism_is_not_constructible(self) -> None:
        with pytest.raises(ValueError, match="layer's formalism"):
            LayersPredictionLens("lofi", "")


def test_a_target_compares_by_tag_and_media_type() -> None:
    assert PublishTarget("hub", "text/markdown") == PublishTarget(
        "hub", "text/markdown"
    )
    assert PublishTarget("hub", "text/markdown") != PublishTarget("hub", "text/plain")
    assert PublishTarget("hub", "text/markdown") != LAYERS_TARGET
    assert len({LAYERS_TARGET, PublishTarget("hub", "text/markdown")}) == 2


def test_a_target_without_a_tag_is_not_constructible() -> None:
    with pytest.raises(ValueError, match="publish target's tag"):
        PublishTarget("", "text/markdown")


def test_a_target_without_a_media_type_is_not_constructible() -> None:
    with pytest.raises(ValueError, match="publish target's media type"):
        PublishTarget("hub", "")


def test_published_values_compare_by_content() -> None:
    assert _analysis() == _analysis()
    assert _analysis() != _analysis("automatic")
    assert len({_analysis(), _analysis()}) == 1


def test_published_values_repr_as_their_constructor_calls() -> None:
    assert repr(_analysis()).startswith(f"PredictedAnalysis({ITEM_ID!r}, ")


def test_importing_the_evaluation_package_does_not_import_the_publish_module() -> None:
    probe = "import sys; import lofi.eval; print('lofi.eval._publish' in sys.modules)"
    finished = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, encoding="utf-8", check=True
    )

    assert finished.stdout.strip() == "False"
