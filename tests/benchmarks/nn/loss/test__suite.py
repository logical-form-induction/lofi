"""Benchmarks the packaged objective suite on its declared accelerator grid."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import pytest
import torch

from lofi._types import ReadoutId
from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.arity import ArityHeadScheme
from lofi.nn.codec import NeuralCodecView
from lofi.nn.decoders import RecursiveTermDecoder, TreeDecoderSettings
from lofi.nn.encoders import TreeRnnEncoder
from lofi.nn.loss._approximation import ApproximationBound
from lofi.nn.loss._mapping import SOFTMAX
from lofi.nn.loss._parser import (
    SPAN_AXES,
    SPLIT_AXES,
    SpanMarginals,
    SplitPosterior,
)
from lofi.nn.loss._readout import (
    CELL_AXES,
    MASS_AXES,
    TermsReadout,
)
from lofi.nn.loss._values import CellChart, ObjectiveOutputs, SpanReadout
from lofi.nn.sites import ComponentId
from lofi.nn.spaces import CombinatorFamily, Concatenate, Space
from lofi.nn.type_directed import UNRESTRICTED_ADMISSION
from lofi.registry import Registry
from lofi.run import plan_training
from lofi.spec import PACKAGED_CONFIG_ROOT, ComponentSpec, RunSpecCodec, derive_seed
from lofi.spec.readouts import DecodedMeaningSettings, ReadoutSource, TermsReadoutSpec
from lofi.spec.readouts.terms import ReadoutMass
from lofi.train import LossAssembly, LossResult, resolve_loss_section

if TYPE_CHECKING:
    from collections.abc import Mapping
    from hashlib import _Hash

    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget
    from torch import Tensor, nn

    from lofi._types import JsonValue
    from lofi.judge import Action, ActionSet

CONFIG: Final = (
    Path(__file__).resolve().parents[4] / "lofi/configs/run/synthetic-parser.yaml"
)
CODEC: Final = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))
BASELINE: Final = (
    Path(__file__).resolve().parents[2]
    / "baselines/Darwin-CPython-3.14-64bit/0007_objectives.json"
)
LENGTHS: Final = (10, 20, 30, 40)
SEMANTIC_LENGTHS: Final = (2, 3)
ITEMS: Final = 32
WIDTH: Final = 64
SEED: Final = 33
MODEL_SEED: Final = 34
DTYPE: Final = torch.float32
SUITE: Final = ("grammaticality", "coherence")
WEIGHTS: Final = (1.0, 0.1)
ROUNDS: Final = 20
WARMUP_ROUNDS: Final = 2


class Batch(NamedTuple):
    """Labels and lengths read by the configured objectives."""

    item_ids: tuple[str, ...]
    lengths: tuple[int, ...]
    labels: tuple[bool, ...]
    step: int = 0


class Harness(NamedTuple):
    """Production objective components and their declared identities."""

    suite: LossAssembly[Batch]
    identity: RecursiveTermDecoder
    application: RecursiveTermDecoder
    space: Space
    readout: TermsReadout
    codec: NeuralCodecView
    actions: ActionSet
    device: torch.device
    loss_declaration: str
    grammar_fingerprint: str
    codec_fingerprint: str
    action_set_fingerprint: str
    module_sha256: str


class RawFixture(NamedTuple):
    """Deterministic host tensors from which each fresh graph is built."""

    spans: SpanIndex
    batch: Batch
    cells: Tensor
    target_log_probabilities: Tensor
    split_log_probabilities: Tensor
    constituent_marginals: Tensor
    rule_marginals: Tensor
    live: Tensor
    built: Tensor
    valid_splits: Tensor
    action_log_probabilities: Mapping[Action, Tensor]
    fixture_sha256: str


class FreshInputs(NamedTuple):
    """One independent objective-boundary graph and its differentiable leaves."""

    outputs: ObjectiveOutputs[Batch]
    cells: Tensor
    target_log_probabilities: Tensor
    split_log_probabilities: Tensor
    constituent_marginals: Tensor
    action_log_probabilities: tuple[Tensor, ...]


def _accelerator() -> torch.device:
    """Return the current accelerator, refusing a processor fallback."""
    device = torch.accelerator.current_accelerator()

    if device is None or not torch.accelerator.is_available():
        pytest.skip("the objective-suite budgets require the current accelerator")

    if device.type == "cpu":
        pytest.skip("the objective-suite budgets do not admit a CPU fallback")

    return device


def _wait(device: torch.device) -> None:
    """Synchronize queued accelerator work when the target has a queue."""
    if device.type != "cpu":
        torch.accelerator.synchronize(device)


def _state_digest(*modules: nn.Module) -> str:
    """Return the digest of ordered module state before device placement."""
    digest = sha256()

    for module in modules:
        digest.update(type(module).__qualname__.encode())

        for name, value in module.state_dict().items():
            digest.update(name.encode())

            if isinstance(value, torch.Tensor):
                _update_tensor(digest, value)
            elif isinstance(value, str):
                digest.update(value.encode())
            else:
                msg = f"module state {name!r} is neither a tensor nor a string"
                raise TypeError(msg)

    return digest.hexdigest()


def _update_tensor(digest: _Hash, tensor: Tensor) -> None:
    """Add one host tensor's exact metadata and bytes to a digest."""
    held = tensor.detach().cpu().contiguous()
    digest.update(str(held.dtype).encode())
    digest.update(json.dumps(tuple(held.shape)).encode())
    digest.update(held.numpy().tobytes())


def _json_settings(value: ComponentSpec) -> dict[str, JsonValue]:
    """Return one validated component declaration without its registry tag."""
    dumped = cast("dict[str, JsonValue]", value.model_dump())
    dumped.pop("kind")

    return dumped


def _harness(device: torch.device) -> Harness:
    """Build the packaged suite with real width-64 components on one device."""
    run_spec = CODEC(CONFIG)
    blueprint = plan_training(run_spec)
    checked_grammar = blueprint.learner.grammar
    language = checked_grammar.language
    category_codec = language.codec.at(language.category_sort)
    codec = NeuralCodecView(category_codec)
    actions = blueprint.actions
    model = run_spec.model
    encoder_spec = model.encoders.get("terms")
    identity_spec = model.decoders.get("identity")
    application_spec = model.decoders.get("application")
    readout_spec = model.readouts.get("category-terms")
    space_spec = model.spaces.get("categories")
    loss = run_spec.loss
    declared_approximation = run_spec.approximation

    if (
        encoder_spec is None
        or identity_spec is None
        or application_spec is None
        or readout_spec is None
        or space_spec is None
        or loss is None
    ):
        msg = "the packaged parser no longer declares its complete default suite"
        raise ValueError(msg)

    if declared_approximation is None:
        msg = "the packaged complete-term readout requires a residual-mass bound"
        raise ValueError(msg)

    approximation = ApproximationBound(declared_approximation.residual_mass)

    encoder_settings = _json_settings(encoder_spec)
    identity_settings = _json_settings(identity_spec)
    application_settings = _json_settings(application_spec)

    if not isinstance(readout_spec, TermsReadoutSpec):
        msg = "the packaged complete-term role is not a term readout"
        raise TypeError(msg)

    readout_settings = readout_spec.settings()
    decoding = readout_settings.decoding

    if decoding is None or decoding.strategy != "exhaustive":
        msg = "the packaged coherence readout no longer enumerates complete terms"
        raise ValueError(msg)

    combinators = space_spec.combinators

    if combinators is None or combinators.merge != "concatenate":
        msg = "the packaged category space no longer concatenates child cells"
        raise ValueError(msg)

    with torch.random.fork_rng():
        torch.manual_seed(MODEL_SEED)
        encoder = TreeRnnEncoder(
            codec=codec,
            choices=language.codec.choices,
            component=ComponentId("terms"),
            scheme=ArityHeadScheme(
                width=WIDTH,
                reached=category_codec.index.signature.operators,
                bound=None,
            ),
            cell=cast("str", encoder_settings["cell"]),
            layers=cast("int", encoder_settings["layers"]),
            norm_cap=cast("float | None", encoder_settings["norm_cap"]),
            inventory="uncached",
        )
        identity = RecursiveTermDecoder(
            codec=codec,
            choices=language.codec.choices,
            admission=UNRESTRICTED_ADMISSION,
            component=ComponentId("identity"),
            settings=TreeDecoderSettings(
                input_width=WIDTH,
                width=cast("int", identity_settings["width"]),
            ),
            choice_heads="flat",
        )
        application = RecursiveTermDecoder(
            codec=codec,
            choices=language.codec.choices,
            admission=UNRESTRICTED_ADMISSION,
            component=ComponentId("application"),
            settings=TreeDecoderSettings(
                input_width=2 * WIDTH,
                width=cast("int", application_settings["width"]),
            ),
            choice_heads="flat",
        )

    module_sha256 = _state_digest(encoder, identity, application)

    for module in (encoder, identity, application):
        module.to(device=device, dtype=DTYPE)

    named_actions = {action.name: action for action in actions.actions}
    family = CombinatorFamily(
        {
            named_actions[action_name]: application
            for action_name in combinators.decoders
        },
        actions=actions,
        rules=checked_grammar.grammar.rules,
        merge=Concatenate(WIDTH),
        codec=category_codec,
        encoder=encoder,
        identity=identity,
        index=language.codec.choices,
        root_sort=space_spec.sort,
    )
    space = Space(
        space_spec.sort,
        codec=codec,
        choices=language.codec.choices,
        store=language.store,
        max_nodes=codec.capacity,
        encoder=encoder,
        identity=identity,
        combinators=family,
    )
    readout = TermsReadout(
        identity,
        codec=codec,
        index=language.codec.choices,
        reads=ReadoutSource.INSIDE,
        decoding=decoding,
        key=ReadoutId("category-terms"),
        targets={},
        strategy=ReadoutMass.FACTORIZED,
        mapping=SOFTMAX,
        store=language.store,
        root_sort=space.sort,
        max_nodes=space.max_nodes,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=approximation,
        run_approximation=approximation,
    )
    suite = resolve_loss_section(
        loss,
        Registry("loss"),
        published=frozenset(
            {
                "actions",
                "batch",
                "cells",
                "chart",
                "marginals",
                "readouts",
                "spans",
                "splits",
            }
        ),
        readouts={"category-terms": readout},
        spaces={"categories": space},
        readout_spaces={"category-terms": space},
        relations={},
        span_decoders={},
        proof_reward=None,
    )
    loss_declaration = json.dumps(
        cast("JsonValue", json.loads(loss.model_dump_json())),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )

    return Harness(
        suite,
        identity,
        application,
        space,
        readout,
        codec,
        actions,
        device,
        loss_declaration,
        blueprint.grammar_fingerprint,
        blueprint.codec_fingerprint,
        blueprint.action_set_fingerprint,
        module_sha256,
    )


@pytest.fixture(scope="module")
def cpu_harness() -> Harness:
    """Return the configured suite used for untimed semantic checks."""
    return _harness(torch.device("cpu"))


@pytest.fixture(scope="module")
def accelerator_harness() -> Harness:
    """Return the configured suite on the accelerator required by the budgets."""
    return _harness(_accelerator())


def _valid_splits(spans: SpanIndex, items: int) -> Tensor:
    """Return the structural split mask repeated over the fixed batch."""
    structural = torch.tensor(spans.split_mask, dtype=torch.bool)

    return structural.unsqueeze(0).expand(items, -1, -1).clone()


def _uniform_tree_marginals(
    spans: SpanIndex,
    valid: Tensor,
    split_log_probabilities: Tensor,
    items: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return exact constituent and rule marginals of the uniform tree posterior."""
    built = torch.ones(items, spans.num_spans, dtype=torch.bool)

    for start, end in spans.leaves():
        built[:, spans.span_id(start, end)] = False

    constituent = torch.zeros(items, spans.num_spans, dtype=DTYPE)
    rule = torch.zeros(items, spans.num_spans, spans.max_splits, dtype=DTYPE)
    constituent[:, spans.span_id(0, spans.n)] = 1.0
    probabilities = torch.where(valid, split_log_probabilities.exp(), 0.0)

    for start, end in spans.spans_by_length_desc(min_length=2):
        span = spans.span_id(start, end)

        for cut in spans.splits(start, end):
            _, column = spans.split_id(start, end, cut)
            share = constituent[:, span] * probabilities[:, span, column]
            rule[:, span, column] = share

            for child_start, child_end in ((start, cut), (cut, end)):
                if child_end - child_start > 1:
                    child = spans.span_id(child_start, child_end)
                    constituent[:, child] += share

    return constituent, rule, built


def _raw(harness: Harness, tokens: int, *, items: int = ITEMS) -> RawFixture:
    """Build one deterministic host fixture at a declared sentence length."""
    spans = SpanIndex(tokens)
    valid = _valid_splits(spans, items)
    counts = valid.sum(dim=-1, keepdim=True).clamp_min(1)
    split_log_probabilities = torch.where(
        valid,
        -counts.to(DTYPE).log(),
        torch.full((), -math.inf, dtype=DTYPE),
    )
    constituent, rule, built = _uniform_tree_marginals(
        spans, valid, split_log_probabilities, items
    )
    generator = torch.Generator(device="cpu").manual_seed(
        derive_seed(SEED, f"objective-suite:n={tokens}")
    )
    cells = torch.randn(items, spans.num_spans, WIDTH, generator=generator, dtype=DTYPE)
    target_log_probabilities = torch.randn(
        items, spans.num_spans, 2, generator=generator, dtype=DTYPE
    ).log_softmax(dim=-1)
    action_share = -math.log(len(harness.actions.actions))
    action_log_probabilities = {
        action: torch.where(
            valid,
            torch.full(valid.shape, action_share, dtype=DTYPE),
            torch.full((), -math.inf, dtype=DTYPE),
        )
        for action in harness.actions.actions
    }
    batch = Batch(
        tuple(f"item-{item}" for item in range(items)),
        (tokens,) * items,
        tuple(item % 2 == 0 for item in range(items)),
    )
    digest = sha256()
    digest.update(harness.loss_declaration.encode())
    digest.update(harness.grammar_fingerprint.encode())
    digest.update(harness.codec_fingerprint.encode())
    digest.update(harness.action_set_fingerprint.encode())
    digest.update(harness.module_sha256.encode())
    digest.update(
        json.dumps(
            {
                "seed": SEED,
                "model_seed": MODEL_SEED,
                "tokens": tokens,
                "items": items,
                "width": WIDTH,
                "dtype": str(DTYPE),
                "actions": [action.name for action in harness.actions.actions],
                "item_ids": batch.item_ids,
                "lengths": batch.lengths,
                "labels": batch.labels,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )

    for tensor in (
        cells,
        target_log_probabilities,
        split_log_probabilities,
        constituent,
        rule,
        live := torch.ones(items, spans.num_spans, dtype=torch.bool),
        built,
        valid,
        *(action_log_probabilities[action] for action in harness.actions.actions),
    ):
        _update_tensor(digest, tensor)

    return RawFixture(
        spans,
        batch,
        cells,
        target_log_probabilities,
        split_log_probabilities,
        constituent,
        rule,
        live,
        built,
        valid,
        action_log_probabilities,
        digest.hexdigest(),
    )


def _leaf(tensor: Tensor, device: torch.device) -> Tensor:
    """Return an independent differentiable copy on the benchmark device."""
    return tensor.to(device).detach().clone().requires_grad_()


def _fresh(harness: Harness, raw: RawFixture) -> FreshInputs:
    """Build a fresh objective-boundary graph from deterministic host tensors."""
    device = harness.device
    items = int(raw.cells.shape[0])
    cells = _leaf(raw.cells, device)
    target_log_probabilities = _leaf(raw.target_log_probabilities, device)
    split_log_probabilities = _leaf(raw.split_log_probabilities, device)
    constituent = _leaf(raw.constituent_marginals, device)
    action_log_probabilities = tuple(
        _leaf(raw.action_log_probabilities[action], device)
        for action in harness.actions.actions
    )
    live = raw.live.to(device)
    built = raw.built.to(device)
    valid = raw.valid_splits.to(device)
    rule = raw.rule_marginals.to(device)
    lengths = torch.full((items,), raw.spans.n, dtype=torch.long, device=device)
    actions = {
        action: LogDistribution(
            value,
            SPLIT_AXES,
            valid,
        )
        for action, value in zip(
            harness.actions.actions, action_log_probabilities, strict=True
        )
    }
    outputs = ObjectiveOutputs(
        spans=raw.spans,
        chart=CellChart(
            LogDistribution(
                torch.zeros(items, raw.spans.num_spans, dtype=DTYPE, device=device),
                SPAN_AXES,
                live,
            )
        ),
        batch=raw.batch,
        splits=SplitPosterior(
            raw.spans,
            LogDistribution(
                split_log_probabilities,
                SPLIT_AXES,
                valid,
                normalized_over="split",
            ),
        ),
        marginals=SpanMarginals(
            raw.spans,
            Encoded(constituent, SPAN_AXES, built),
            Encoded(rule, SPLIT_AXES, valid),
            lengths,
        ),
        actions=actions,
        readouts={
            ReadoutId("sentences"): SpanReadout(
                raw.spans,
                LogDistribution(
                    target_log_probabilities,
                    MASS_AXES,
                    live,
                    normalized_over="outcome",
                ),
            ),
        },
        cells=Encoded(cells, CELL_AXES, live),
    )

    return FreshInputs(
        outputs,
        cells,
        target_log_probabilities,
        split_log_probabilities,
        constituent,
        action_log_probabilities,
    )


def _zero_grad(harness: Harness) -> None:
    """Clear every production module gradient before a fresh backward pass."""
    harness.identity.zero_grad(set_to_none=True)
    harness.application.zero_grad(set_to_none=True)


def _timed_forward(
    suite: LossAssembly[Batch], outputs: ObjectiveOutputs[Batch], device: torch.device
) -> Tensor:
    """Evaluate the suite and wait for its queued work."""
    _wait(device)
    total = suite.evaluate(outputs, step=0).total
    _wait(device)

    return total.detach()


def _forward_setup(
    harness: Harness, raw: RawFixture
) -> tuple[
    tuple[LossAssembly[Batch], ObjectiveOutputs[Batch], torch.device], dict[str, object]
]:
    """Return one fresh forward graph outside the timed callable."""
    fresh = _fresh(harness, raw)
    _wait(harness.device)

    return (harness.suite, fresh.outputs, harness.device), {}


def _timed_backward(total: Tensor, device: torch.device) -> None:
    """Differentiate one pre-evaluated total and wait for queued work."""
    _wait(device)
    total.backward()
    _wait(device)


def _backward_setup(
    harness: Harness, raw: RawFixture
) -> tuple[tuple[Tensor, torch.device], dict[str, object]]:
    """Evaluate a fresh graph outside the timed backward callable."""
    _zero_grad(harness)
    fresh = _fresh(harness, raw)
    total = harness.suite.evaluate(fresh.outputs, step=0).total
    _wait(harness.device)

    return (total, harness.device), {}


def _record(
    benchmark: BenchmarkFixture,
    harness: Harness,
    raw: RawFixture,
    *,
    budget: str,
    phase: str,
) -> float:
    """Record the complete benchmark identity and return the measured mean."""
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the objective suite ran but pytest-benchmark timed nothing")

    mean_ms = measured.stats.mean * 1e3
    median_ms = measured.stats.median * 1e3
    benchmark.extra_info["budget"] = budget
    benchmark.extra_info["phase"] = phase
    benchmark.extra_info["tokens"] = raw.spans.n
    benchmark.extra_info["items"] = ITEMS
    benchmark.extra_info["width"] = WIDTH
    benchmark.extra_info["dtype"] = str(DTYPE)
    benchmark.extra_info["device"] = str(harness.device)
    benchmark.extra_info["suite"] = list(harness.suite.names)
    benchmark.extra_info["weights"] = list(harness.suite.weights)
    benchmark.extra_info["fixture_sha256"] = raw.fixture_sha256
    benchmark.extra_info["loss_declaration"] = harness.loss_declaration
    benchmark.extra_info["grammar_fingerprint"] = harness.grammar_fingerprint
    benchmark.extra_info["codec_fingerprint"] = harness.codec_fingerprint
    benchmark.extra_info["action_set_fingerprint"] = harness.action_set_fingerprint
    benchmark.extra_info["mean_ms"] = mean_ms
    benchmark.extra_info["median_ms"] = median_ms

    return mean_ms


def _json_object(value: object, description: str) -> dict[str, JsonValue]:
    """Return a JSON object or fail with the baseline location named."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        pytest.fail(f"{description} in {BASELINE} is not a JSON object")

    return cast("dict[str, JsonValue]", value)


def _baseline_entries() -> tuple[dict[str, JsonValue], ...]:
    """Read the exact saved objective baseline entries."""
    document = _json_object(
        json.loads(BASELINE.read_text(encoding="utf-8")), "the saved run"
    )
    benchmarks = document.get("benchmarks")

    if not isinstance(benchmarks, list):
        pytest.fail(f"benchmarks in {BASELINE} is not a JSON array")

    return tuple(
        _json_object(entry, f"benchmark entry {index}")
        for index, entry in enumerate(benchmarks)
    )


def test_accelerator_selection_refuses_a_processor_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check that timed budgets cannot silently run on a processor."""
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)

    with pytest.raises(pytest.skip.Exception, match="do not admit a CPU fallback"):
        _accelerator()


def test_the_packaged_suite_and_fixture_have_the_declared_structure(
    cpu_harness: Harness,
) -> None:
    """Check the suite, grid, precision and production component identities."""
    raw = _raw(cpu_harness, LENGTHS[0])
    fresh = _fresh(cpu_harness, raw)
    outputs = fresh.outputs

    assert cpu_harness.device == torch.device("cpu")
    assert cpu_harness.suite.names == SUITE
    assert cpu_harness.suite.weights == WEIGHTS
    assert LENGTHS == (10, 20, 30, 40)
    assert tuple(action.name for action in cpu_harness.actions.actions) == (
        "identity",
        "application:/",
        "application:\\",
    )
    assert cpu_harness.space.sort == "Cat"
    assert cpu_harness.space.max_nodes == cpu_harness.codec.capacity
    assert cpu_harness.space.encoder.output_width == WIDTH
    assert cpu_harness.identity.input_width == WIDTH
    assert cpu_harness.application.input_width == 2 * WIDTH
    assert cpu_harness.space.combinators is not None
    assert set(cpu_harness.space.combinators.decoders.values()) == {
        cpu_harness.application
    }
    assert cpu_harness.readout.key == ReadoutId("category-terms")
    assert cpu_harness.readout.decoding is not None
    assert cpu_harness.readout.decoding.strategy == "exhaustive"
    assert fresh.cells.shape == (ITEMS, raw.spans.num_spans, WIDTH)
    assert fresh.target_log_probabilities.shape == (
        ITEMS,
        raw.spans.num_spans,
        2,
    )
    assert fresh.split_log_probabilities.shape == (
        ITEMS,
        raw.spans.num_spans,
        raw.spans.max_splits,
    )
    assert all(tensor.dtype is DTYPE for tensor in fresh[1:5])
    assert all(tensor.device.type == cpu_harness.device.type for tensor in fresh[1:5])
    assert outputs.readouts[ReadoutId("sentences")].computed.all()


def test_the_fixture_digest_is_deterministic_and_shape_sensitive(
    cpu_harness: Harness,
) -> None:
    """Check that declaration, seed and raw inputs determine the fixture identity."""
    rebuilt = _harness(torch.device("cpu"))
    fixtures = tuple(_raw(cpu_harness, tokens) for tokens in LENGTHS)
    repeated = tuple(_raw(rebuilt, tokens) for tokens in LENGTHS)

    assert cpu_harness.module_sha256 == rebuilt.module_sha256
    assert len({fixture.fixture_sha256 for fixture in fixtures}) == len(LENGTHS)

    for tokens, fixture, again in zip(LENGTHS, fixtures, repeated, strict=True):
        spans = tokens * (tokens + 1) // 2

        assert fixture.fixture_sha256 == again.fixture_sha256
        assert torch.equal(fixture.cells, again.cells)
        assert fixture.cells.shape == (ITEMS, spans, WIDTH)
        assert fixture.split_log_probabilities.shape == (
            ITEMS,
            spans,
            tokens - 1,
        )


def test_the_saved_objective_baseline_matches_the_live_contract(
    cpu_harness: Harness,
) -> None:
    """Check that the saved objective run is exact and reproducible."""
    if not BASELINE.is_file():
        pytest.skip(f"the saved objective baseline is not yet present at {BASELINE}")

    expected: dict[str, tuple[str, int, str, str]] = {}
    source = "tests/benchmarks/nn/loss/test__suite.py"

    for test_name, phase, budget in (
        (
            "test_the_default_suite_forward_meets_its_budget",
            "forward",
            "objective-forward",
        ),
        (
            "test_the_default_suite_backward_meets_its_budget",
            "backward",
            "objective-backward",
        ),
    ):
        for tokens in LENGTHS:
            name = f"{test_name}[{tokens}-tokens]"
            expected[f"{source}::{name}"] = (name, tokens, phase, budget)

    entries = _baseline_entries()
    observed: dict[str, dict[str, JsonValue]] = {}

    for entry in entries:
        fullname = entry.get("fullname")
        assert isinstance(fullname, str)
        observed[fullname] = entry

    assert len(entries) == len(observed) == 8
    assert set(observed) == set(expected)
    fixture_sha256 = {
        tokens: _raw(cpu_harness, tokens).fixture_sha256 for tokens in LENGTHS
    }

    for fullname, (name, tokens, phase, budget) in expected.items():
        entry = observed[fullname]
        params = _json_object(entry.get("params"), f"params for {fullname}")
        extra = _json_object(entry.get("extra_info"), f"extra_info for {fullname}")
        stats = _json_object(entry.get("stats"), f"stats for {fullname}")
        mean = stats.get("mean")
        mean_ms = extra.get("mean_ms")

        assert entry.get("name") == name
        assert entry.get("param") == f"{tokens}-tokens"
        assert params == {"tokens": tokens}
        assert entry.get("group") == "default-objective-suite"
        assert extra.get("phase") == phase
        assert extra.get("budget") == budget
        assert stats.get("rounds") == ROUNDS
        assert stats.get("iterations") == 1
        assert isinstance(mean, int | float)
        assert not isinstance(mean, bool)
        assert isinstance(mean_ms, int | float)
        assert not isinstance(mean_ms, bool)
        assert float(mean) * 1e3 == float(mean_ms)
        assert extra.get("suite") == list(SUITE)
        assert extra.get("weights") == list(WEIGHTS)
        assert extra.get("loss_declaration") == cpu_harness.loss_declaration
        assert extra.get("grammar_fingerprint") == cpu_harness.grammar_fingerprint
        assert extra.get("codec_fingerprint") == cpu_harness.codec_fingerprint
        assert extra.get("action_set_fingerprint") == cpu_harness.action_set_fingerprint
        assert extra.get("fixture_sha256") == fixture_sha256[tokens]


def test_the_suite_returns_finite_values_and_reaches_its_inputs(
    cpu_harness: Harness,
) -> None:
    """Check finite semantics and nonzero finite gradients outside timed bodies."""
    raw = _raw(cpu_harness, SEMANTIC_LENGTHS[0], items=1)
    fresh = _fresh(cpu_harness, raw)
    _zero_grad(cpu_harness)
    result = cpu_harness.suite.evaluate(fresh.outputs, step=0)
    result.total.backward()
    _wait(cpu_harness.device)

    assert isinstance(result, LossResult)
    assert torch.isfinite(result.total)
    assert tuple(record.name for record in result.terms) == SUITE
    assert all(torch.isfinite(record.term.value) for record in result.terms)
    assert all(
        torch.isfinite(value)
        for record in result.terms
        for value in record.term.diagnostics.values()
    )

    for tensor in (
        fresh.cells,
        fresh.target_log_probabilities,
        fresh.split_log_probabilities,
        fresh.constituent_marginals,
    ):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert bool(tensor.grad.abs().sum() > 0.0)

    identity, *composing = fresh.action_log_probabilities
    assert identity.grad is None

    # Both composing actions use the same decoder, so their posterior
    # reweighting is constant in the packaged suite.
    for tensor in composing:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.equal(tensor.grad, torch.zeros_like(tensor.grad))

    identity_gradients = tuple(
        parameter.grad
        for parameter in cpu_harness.identity.parameters()
        if parameter.grad is not None
    )
    application_gradients = tuple(
        parameter.grad
        for parameter in cpu_harness.application.parameters()
        if parameter.grad is not None
    )

    for gradients in (identity_gradients, application_gradients):
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(bool(gradient.abs().sum() > 0.0) for gradient in gradients)


def test_the_suite_value_depends_on_sentence_length(cpu_harness: Harness) -> None:
    """Check that token length changes the real coherence computation."""
    short = cpu_harness.suite.evaluate(
        _fresh(cpu_harness, _raw(cpu_harness, SEMANTIC_LENGTHS[0], items=1)).outputs,
        step=0,
    )
    long = cpu_harness.suite.evaluate(
        _fresh(cpu_harness, _raw(cpu_harness, SEMANTIC_LENGTHS[1], items=1)).outputs,
        step=0,
    )
    _wait(cpu_harness.device)

    short_coherence = short.terms[1].term.value
    long_coherence = long.terms[1].term.value

    assert short_coherence.shape == long_coherence.shape == torch.Size([])
    assert not torch.equal(short_coherence, long_coherence)


@pytest.mark.benchmark(group="default-objective-suite")
@pytest.mark.parametrize(
    "tokens", LENGTHS, ids=("10-tokens", "20-tokens", "30-tokens", "40-tokens")
)
def test_the_default_suite_forward_meets_its_budget(
    benchmark: BenchmarkFixture,
    accelerator_harness: Harness,
    tokens: int,
    wall_clock_budget: WallClockBudget,
) -> None:
    """Measure only the default suite's forward evaluation."""
    raw = _raw(accelerator_harness, tokens)
    benchmark.pedantic(
        _timed_forward,
        setup=lambda: _forward_setup(accelerator_harness, raw),
        rounds=ROUNDS,
        warmup_rounds=WARMUP_ROUNDS,
        iterations=1,
    )
    mean_ms = _record(
        benchmark,
        accelerator_harness,
        raw,
        budget="objective-forward",
        phase="forward",
    )

    if wall_clock_budget.enforced:
        match tokens:
            case 10:
                assert mean_ms <= 250.0
            case 20:
                assert mean_ms <= 900.0
            case 30:
                assert mean_ms <= 2_700.0
            case 40:
                assert mean_ms <= 6_300.0
            case unreachable:
                pytest.fail(
                    f"no objective-forward budget is declared for {unreachable} tokens"
                )


@pytest.mark.benchmark(group="default-objective-suite")
@pytest.mark.parametrize(
    "tokens", LENGTHS, ids=("10-tokens", "20-tokens", "30-tokens", "40-tokens")
)
def test_the_default_suite_backward_meets_its_budget(
    benchmark: BenchmarkFixture,
    accelerator_harness: Harness,
    tokens: int,
    wall_clock_budget: WallClockBudget,
) -> None:
    """Measure only backward over a freshly evaluated suite graph."""
    raw = _raw(accelerator_harness, tokens)
    benchmark.pedantic(
        _timed_backward,
        setup=lambda: _backward_setup(accelerator_harness, raw),
        rounds=ROUNDS,
        warmup_rounds=WARMUP_ROUNDS,
        iterations=1,
    )
    mean_ms = _record(
        benchmark,
        accelerator_harness,
        raw,
        budget="objective-backward",
        phase="backward",
    )

    if wall_clock_budget.enforced:
        match tokens:
            case 10:
                assert mean_ms <= 250.0
            case 20:
                assert mean_ms <= 1_350.0
            case 30:
                assert mean_ms <= 4_600.0
            case 40:
                assert mean_ms <= 11_200.0
            case unreachable:
                pytest.fail(
                    f"no objective-backward budget is declared for {unreachable} tokens"
                )
