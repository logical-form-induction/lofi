"""One declaration trains combinator decoders over types and logical forms."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from lofi.interpret import run_action
from lofi.nn._values import Encoded
from lofi.nn.strategies import greedy
from lofi.run import RandomStreams, SeedPolicy, plan_training, resolve_device_policy
from lofi.spec import RunSpecCodec
from lofi.train import Draw, StepFittingPlan
from lofi.train._assembly import prepare_training

if TYPE_CHECKING:
    from lofi.data.synthetic import InterpretationFunction
    from lofi.nn.ports import FrontierDecoder, TermEncoder
    from lofi.spec.grammar import GrammarAssembly
    from lofi.spec.language import LanguageAssembly
    from lofi.terms import Term
    from lofi.train import PreparedTraining


DOCUMENT = Path(__file__).with_name("fixtures") / "types-and-forms.yaml"


def _prepared() -> tuple[
    PreparedTraining, GrammarAssembly, InterpretationFunction | None
]:
    """Compose and prepare the document through the current run boundary."""
    run = RunSpecCodec()(DOCUMENT)
    blueprint = plan_training(run)
    streams = RandomStreams(SeedPolicy(29, samplers=blueprint.sampler_streams))
    streams.apply()

    return (
        prepare_training(blueprint, streams, resolve_device_policy(run)),
        blueprint.learner.grammar,
        blueprint.learner.interpretation,
    )


def _pair(
    interpretation: InterpretationFunction,
    *,
    functor: Term,
    argument: Term,
) -> tuple[Term, Term, Term]:
    """Return two lexical meanings and their beta-normal application."""
    terms = interpretation.actions.resolve("application:/", None)

    if len(terms) != 1:
        msg = "the fixture did not derive forward application"
        raise AssertionError(msg)

    (action,) = terms
    output = run_action(
        action,
        (functor, argument),
        calculus=interpretation.calculus,
        normalization=interpretation.normalization,
    )

    return functor, argument, output


def _decode(
    prepared: PreparedTraining,
    language: LanguageAssembly,
    *,
    encoder_role: str,
    decoder_role: str,
    sort: str,
    inputs: tuple[Term, Term],
) -> Term:
    """Greedily decode one parent from two represented daughter terms."""
    encoder = cast("TermEncoder", prepared.model.modules[encoder_role])
    decoder = cast("FrontierDecoder[object]", prepared.model.modules[decoder_role])
    codec = language.codec.at(sort)
    left = encoder.encode(codec.encode_batch((inputs[0],)))
    right = encoder.encode(codec.encode_batch((inputs[1],)))
    merged = Encoded(
        torch.cat((left.tensor, right.tensor), dim=-1),
        left.axes,
        left.mask,
    )
    decoded = greedy(
        decoder,
        merged,
        index=language.codec.choices,
        store=language.store,
        root_sort=sort,
        max_nodes=codec.capacity,
    )

    if len(decoded.terms) != 1:
        msg = f"greedy decoding returned {len(decoded.terms)} terms"
        raise AssertionError(msg)

    return decoded.terms[0]


def test_one_document_trains_type_and_form_combinators_to_exact_match() -> None:
    prepared, grammar, interpretation = _prepared()
    plan = cast("StepFittingPlan", prepared.plan)

    for step in range(plan.steps):
        seed = 10_000 + step
        plan.trainer_at(step).run(
            Draw(seed, plan.batch_size, step=step, horizon=plan.steps)
        )

    language = grammar.language

    if interpretation is None:
        msg = "the fixture did not lower its declared interpretation"
        raise AssertionError(msg)

    category_functor = grammar.syntax.parse(grammar.store, "S/NP")
    category_argument = grammar.syntax.parse(grammar.store, "NP")
    type_pair = (
        interpretation.erasure(category_functor),
        interpretation.erasure(category_argument),
        interpretation.erasure(grammar.syntax.parse(grammar.store, "S")),
    )
    functor = interpretation.primary("predicate", category_functor).meaning
    argument = interpretation.primary("argument", category_argument).meaning

    if functor is None or argument is None:
        msg = "the fixture lexical entries did not lower their declared meanings"
        raise AssertionError(msg)

    form_pair = _pair(
        interpretation,
        functor=functor,
        argument=argument,
    )

    decoded_type = _decode(
        prepared,
        language,
        encoder_role="semantic-types",
        decoder_role="type-application",
        sort=language.type_sort,
        inputs=type_pair[:2],
    )
    decoded_form = _decode(
        prepared,
        language,
        encoder_role="logical-forms",
        decoder_role="form-application",
        sort=language.term_sort,
        inputs=form_pair[:2],
    )

    assert decoded_type == type_pair[2]
    assert decoded_form == form_pair[2]
