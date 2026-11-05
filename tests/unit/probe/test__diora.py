from __future__ import annotations

from hashlib import sha256
from math import inf
from typing import TYPE_CHECKING, Final, override

import pytest
import torch
from torch import nn

from lofi._artifacts import ArtifactFetcher, PinnedArchive
from lofi.deduce import SpanIndex
from lofi.nn.sites import Role
from lofi.probe._boundary import ExternalModelSpec
from lofi.probe._diagnose import DiagnosticContext
from lofi.probe._diora import (
    DioraCheckpointLoader,
    DioraForward,
    SplitPosteriorKL,
    diora_model_spec,
)
from lofi.probe._values import Observation, Trace
from lofi.spec.artifacts import UrlSourceSpec

if TYPE_CHECKING:
    from pathlib import Path

ARTIFACT: Final = b"released checkpoint"
DIGEST: Final = sha256(ARTIFACT).hexdigest()
INDEX: Final = SpanIndex(4)


MODEL: Final = "diora-test"


def a_release(*, digest: str = DIGEST) -> UrlSourceSpec:
    return UrlSourceSpec(
        url="https://example.test/diora-checkpoints.zip", sha256=digest
    )


class _Archives:
    def __init__(self, payload: bytes = ARTIFACT) -> None:
        self.payload = payload
        self.requests: list[str] = []

    def download(self, url: str, target: Path, /) -> None:
        self.requests.append(url)
        target.write_bytes(self.payload)


class _Snapshots:
    def snapshot(self, repository: str, revision: str, target: Path, /) -> None:
        raise AssertionError((repository, revision, target))


def a_fetcher(
    root: Path, archives: _Archives | None = None, *, offline: bool = False
) -> ArtifactFetcher:
    return ArtifactFetcher(
        root,
        archives=_Archives() if archives is None else archives,
        snapshots=_Snapshots(),
        offline=offline,
    )


class _Backend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Identity()
        self.inside = nn.Identity()
        self.outside = nn.Identity()

    @property
    def input_width(self) -> int:
        return 2

    @override
    def forward(self, tokens: torch.Tensor) -> DioraForward:
        items = int(tokens.shape[0])
        inside = tokens.new_zeros((items, INDEX.num_spans), dtype=torch.float64)
        outside = tokens.new_zeros((items, INDEX.num_spans), dtype=torch.float64)
        alpha = tokens.new_full(
            (items, INDEX.num_spans, INDEX.max_splits),
            -inf,
            dtype=torch.float64,
        )

        for start, end in INDEX.spans_by_length(min_length=2):
            choices = end - start - 1
            alpha[:, INDEX.span_id(start, end), :choices] = -torch.log(
                torch.tensor(float(choices), dtype=torch.float64)
            )

        return DioraForward(
            tokens,
            self.alpha(alpha),
            self.inside(inside),
            self.outside(outside),
        )


class _Factory:
    def __init__(self) -> None:
        self.requests: list[Path] = []

    def build(self, checkpoint: Path, /) -> nn.Module:
        self.requests.append(checkpoint)
        return _Backend()


def a_trace(*observations: Observation) -> Trace:
    trace = Trace()

    for observation in observations:
        trace.record(observation)

    trace.close()
    return trace


def chart_observations(
    *,
    lengths: tuple[int, ...] = (4,),
    changed_attention: bool = False,
) -> tuple[Observation, Observation, Observation]:
    release = a_release()
    declared = diora_model_spec(MODEL, release)
    sites = {site.role: site for site in declared.sites_runtime()}
    alpha_site = sites[Role.SPLIT_DISTRIBUTION]
    inside_site = sites[Role.INSIDE_SCORE]
    outside_site = sites[Role.OUTSIDE_SCORE]
    alpha = torch.full(
        (len(lengths), INDEX.num_spans, INDEX.max_splits),
        -inf,
        dtype=torch.float64,
    )
    split_mask = torch.zeros_like(alpha, dtype=torch.bool)
    inside = torch.zeros(len(lengths), INDEX.num_spans, dtype=torch.float64)
    span_mask = torch.zeros_like(inside, dtype=torch.bool)

    for item, length in enumerate(lengths):
        for start, end in INDEX.spans_by_length():
            if end > length:
                continue

            span = INDEX.span_id(start, end)
            span_mask[item, span] = True

            if end - start < 2:
                continue

            choices = end - start - 1
            alpha[item, span, :choices] = -torch.log(
                torch.tensor(float(choices), dtype=torch.float64)
            )
            split_mask[item, span, :choices] = True

    if changed_attention:
        row = INDEX.span_id(0, 3)
        alpha[0, row, :2] = torch.tensor((0.8, 0.2), dtype=torch.float64).log()
        inside[0, INDEX.span_id(0, 1)] = 1.0

    return (
        Observation(
            alpha_site,
            alpha,
            axes=("item", "span", "split"),
            mask=split_mask,
        ),
        Observation(
            inside_site,
            inside,
            axes=("item", "span"),
            mask=span_mask,
        ),
        Observation(
            outside_site,
            torch.zeros_like(inside),
            axes=("item", "span"),
            mask=span_mask,
        ),
    )


def test_the_release_is_declared_on_the_external_model_boundary() -> None:
    release = a_release()
    declared = diora_model_spec(MODEL, release)
    encoded = declared.model_dump()

    assert ExternalModelSpec.model_validate(encoded) == declared
    assert declared.checkpoint == release
    assert declared.site_map().checkpoint == PinnedArchive(release.url, DIGEST)
    assert {site.role for site in declared.sites_runtime()} == {
        Role.SPLIT_DISTRIBUTION,
        Role.INSIDE_SCORE,
        Role.OUTSIDE_SCORE,
    }


def test_a_release_is_fetched_once_and_then_loads_offline(
    tmp_path: Path,
) -> None:
    archives = _Archives()
    factory = _Factory()
    declared = diora_model_spec(MODEL, a_release()).site_map()

    first = DioraCheckpointLoader(a_fetcher(tmp_path, archives), factory).load(declared)
    second = DioraCheckpointLoader(
        a_fetcher(tmp_path, archives, offline=True), factory
    ).load(declared)

    assert len(archives.requests) == 1
    assert len(factory.requests) == 2
    assert factory.requests[0] == factory.requests[1]
    assert first.model.site_map() == second.model.site_map()
    assert first.input_width == second.input_width == 2
    assert first.model.declared.checkpoint == PinnedArchive(a_release().url, DIGEST)


def test_offline_loading_refuses_an_absent_release(tmp_path: Path) -> None:
    loader = DioraCheckpointLoader(a_fetcher(tmp_path, offline=True), _Factory())

    with pytest.raises(FileNotFoundError, match="offline"):
        loader.load(diora_model_spec(MODEL, a_release()).site_map())


def test_a_download_whose_digest_differs_is_refused(tmp_path: Path) -> None:
    loader = DioraCheckpointLoader(
        a_fetcher(tmp_path, _Archives(b"not the checkpoint")), _Factory()
    )

    with pytest.raises(ValueError, match="SHA-256"):
        loader.load(diora_model_spec(MODEL, a_release()).site_map())


def test_a_released_model_declares_its_checkpoint(tmp_path: Path) -> None:
    declared = diora_model_spec(MODEL, a_release()).with_(checkpoint=None)
    loader = DioraCheckpointLoader(a_fetcher(tmp_path), _Factory())

    with pytest.raises(ValueError, match="declares its checkpoint"):
        loader.load(declared.site_map())


def test_the_adapter_publishes_all_three_chart_sites(tmp_path: Path) -> None:
    loaded = DioraCheckpointLoader(a_fetcher(tmp_path), _Factory()).load(
        diora_model_spec(MODEL, a_release()).site_map()
    )
    external = loaded.model
    sites = {site.role: site for site in external.site_map()}
    observed = frozenset(sites.values())

    with external.instrument(observed, patches={}) as scope:
        external.run({"tokens": torch.ones(1, INDEX.n)})

    assert set(scope.readings) == observed


def test_attention_equal_to_the_true_split_posterior_has_zero_kl() -> None:
    alpha, inside, outside = chart_observations(lengths=(3, 4))
    published = diora_model_spec(MODEL, a_release()).site_map().published
    result = SplitPosteriorKL().run(
        a_trace(alpha, inside, outside),
        context=DiagnosticContext(published, span_index=INDEX),
    )

    assert result.value(alpha.site, "kl:length:3") == pytest.approx(0.0, abs=1e-12)
    assert result.value(alpha.site, "kl:length:4") == pytest.approx(0.0, abs=1e-12)


def test_attention_that_omits_child_inside_mass_has_positive_kl() -> None:
    alpha, inside, outside = chart_observations(changed_attention=True)
    published = diora_model_spec(MODEL, a_release()).site_map().published
    result = SplitPosteriorKL().run(
        a_trace(alpha, inside, outside),
        context=DiagnosticContext(published, span_index=INDEX),
    )

    assert result.value(alpha.site, "kl:length:4") > 0.0


def test_the_registered_diagnostic_declares_the_complete_chart_contract() -> None:
    assert SplitPosteriorKL.tag == "split-posterior-kl"
    assert SplitPosteriorKL.requires == frozenset(
        {Role.SPLIT_DISTRIBUTION, Role.INSIDE_SCORE, Role.OUTSIDE_SCORE}
    )
