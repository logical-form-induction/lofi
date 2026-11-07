"""Synthetic inference publishes forms through the learner language's one index."""

from pathlib import Path
from typing import TYPE_CHECKING, Final

import torch

from lofi._types import ComponentId
from lofi.data.synthetic.spec.inference import (
    SyntheticInferenceDecl,
    SyntheticInferenceLowerer,
)
from lofi.data.synthetic.spec.interpretation import lower_interpretation
from lofi.deduce import ParsePolicy, Reachability, Retain, SpanIndex
from lofi.judge import ActionSet
from lofi.nn._values import Encoded
from lofi.nn.build import BuildContext
from lofi.nn.codec import NeuralCodecView
from lofi.nn.loss._approximation import ApproximationBound
from lofi.nn.loss._readout import CELL_AXES
from lofi.nn.policy import DevicePolicy
from lofi.nn.tokens.embedder import ReservedTokens, TokenVocabulary
from lofi.spec import ModelSection, SpecLoader
from lofi.spec._plugins import load_plugins
from lofi.spec.grammar import GrammarIso
from lofi.spec.language import LanguageIso
from lofi.train._spaces import wire_spaces

if TYPE_CHECKING:
    from lofi.data.synthetic import InterpretationFunction, SyntheticInference
    from lofi.nn.forms import WeightedTerms
    from lofi.spec.grammar import GrammarAssembly
    from lofi.spec.language import LanguageAssembly
    from lofi.terms import Term
    from lofi.train._spaces import SpaceGraph

DOCUMENTS: Final = Path(__file__).with_name("documents")
INFERENCE: Final = DOCUMENTS / "higher-order-inference-forms.yaml"
MODEL: Final = DOCUMENTS / "model" / "higher-order-inference-forms.yaml"
SYNTHETIC_INFERENCE: Final = (
    Path(__file__).parents[2]
    / "lofi"
    / "configs"
    / "data"
    / "datasets"
    / "train"
    / "dataset"
    / "synthetic-inference.yaml"
)
FOREST_POLICY: Final = ParsePolicy(Retain.FOREST, Reachability.ALL)


def _lowered() -> tuple[GrammarAssembly, InterpretationFunction, SyntheticInference]:
    """Lower the inference document through its declared grammar and language."""
    declaration = (
        SpecLoader().compose_traced(INFERENCE, schema=SyntheticInferenceDecl).spec
    )
    drawn = SyntheticInferenceLowerer().lower(declaration, FOREST_POLICY)
    grammar, interpretation = lower_interpretation(
        declaration.grammar,
        declaration.source.language,
        declaration.source.interpretation,
        GrammarIso(),
        LanguageIso(),
    )

    return grammar, interpretation, drawn


def _categories(grammar: GrammarAssembly) -> tuple[Term, ...]:
    """Return the declared category representatives in first-use order."""
    candidates = (
        *(
            entry.category
            for form in grammar.lexicon.forms
            for entry in grammar.lexicon.entries(form)
        ),
        *grammar.grammar.targets.categories,
    )

    return tuple(dict.fromkeys(candidates))


def _model() -> ModelSection:
    """Compose the forms probe whose exhaustive bound comes from the run."""
    load_plugins()

    return (
        SpecLoader(search_path=(DOCUMENTS,))
        .compose_traced(
            MODEL,
            schema=ModelSection,
            extra={"language": {"from": "data"}},
        )
        .spec
    )


def _decode(
    model: ModelSection,
    grammar: GrammarAssembly,
    interpretation: InterpretationFunction,
    residual_mass: float,
) -> tuple[WeightedTerms, SpaceGraph, LanguageAssembly]:
    """Build both spaces and decode one form-valued cell."""
    language = grammar.language
    actions = ActionSet.derive(
        (rule.name for rule in grammar.grammar.rules),
        identity=model.actions.identity,
        null=model.actions.null,
    )
    context = BuildContext(
        component=ComponentId("model"),
        policy=DevicePolicy(torch.device("cpu"), torch.float32),
        language=language,
        codec=NeuralCodecView(language.codec.at(language.term_sort)),
        vocabulary=TokenVocabulary(
            tuple(grammar.lexicon.forms),
            reserved=ReservedTokens({"<pad>": 0, "<unk>": 1}),
            unknown="<unk>",
        ),
        actions=actions,
        rules=grammar.grammar.rules,
        categories=_categories(grammar),
        producers={},
        targets={"grammar-goal": grammar.grammar.targets},
        loader=None,
        approximation=ApproximationBound(residual_mass),
    )
    graph = wire_spaces(
        model,
        context,
        parser=None,
        pretrained_resources={},
        erasure=interpretation.erasure,
    )
    spans = SpanIndex(1)
    width = graph.spaces["forms"].encoder.output_width
    decoded = graph.readouts["published-forms"].decode(
        Encoded(torch.zeros(1, 1, width), CELL_AXES),
        spans=spans,
    )

    return decoded.at(0, spans.span_id(0, 1)), graph, language


def test_higher_order_inference_publishes_forms_with_mass_evidence() -> None:
    grammar, interpretation, drawn = _lowered()
    gold = tuple(
        interpretation.read_text(form.text)
        for pair in drawn.pairs
        for form in (pair.premise, pair.hypothesis)
    )

    assert gold
    assert all(
        interpretation.language.print_type(interpretation.language.type_of(form)) == "t"
        for form in gold
    )

    torch.manual_seed(19)
    loose, loose_graph, language = _decode(
        _model(), grammar, interpretation, residual_mass=0.9
    )
    torch.manual_seed(19)
    tight, tight_graph, _ = _decode(
        _model(), grammar, interpretation, residual_mass=0.0
    )

    assert loose_graph.spaces["types"].choices is language.codec.choices
    assert loose_graph.spaces["forms"].choices is language.codec.choices
    assert tight_graph.spaces["forms"].choices is language.codec.choices
    assert loose.terms
    assert tight.terms
    assert all(
        term.op.sort == language.term_sort for term in (*loose.terms, *tight.terms)
    )
    assert all(language.store.contains(term) for term in (*loose.terms, *tight.terms))
    loose_residual = float(loose.log_residual.detach().exp())
    tight_residual = float(tight.log_residual.detach().exp())
    assert 0.0 <= tight_residual < loose_residual <= 1.0


def test_countermodel_labeling_preserves_labels_and_spends_fewer_prover_calls() -> None:
    loader = SpecLoader()
    filtered_decl = loader.compose_traced(
        SYNTHETIC_INFERENCE, schema=SyntheticInferenceDecl
    ).spec
    plain_decl = loader.compose_traced(
        SYNTHETIC_INFERENCE,
        schema=SyntheticInferenceDecl,
        overrides=("source.interpretation.countermodels=null",),
    ).spec
    filtered, filtered_interpretation = (
        SyntheticInferenceLowerer().lower_with_interpretation(
            filtered_decl, FOREST_POLICY
        )
    )
    plain, plain_interpretation = SyntheticInferenceLowerer().lower_with_interpretation(
        plain_decl, FOREST_POLICY
    )

    try:
        filtered_rows = tuple(
            (pair.premise.text, pair.hypothesis.text, pair.label)
            for pair in filtered.pairs
        )
        plain_rows = tuple(
            (pair.premise.text, pair.hypothesis.text, pair.label)
            for pair in plain.pairs
        )

        assert filtered_rows == plain_rows
        assert (
            filtered_interpretation.service.report.calls
            < plain_interpretation.service.report.calls
        )
    finally:
        filtered_interpretation.service.close()
        plain_interpretation.service.close()
