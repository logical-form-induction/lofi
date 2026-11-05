"""Instrument a tiny third-party Torch model through a declared site map."""

import sys
from typing import Final, NamedTuple, override

import torch
from torch import Tensor, nn

from lofi.nn.sites import ComponentId, Role, Site
from lofi.probe import (
    ExternalModel,
    ExternalModelSpec,
    ExternalSiteDecl,
    ExternalSiteMapIso,
    Output,
    Zero,
    instrument,
)


class _TupleLayer(nn.Module):
    """A hidden layer whose first output is the published tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(2))

    @override
    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Return a hidden value and an auxiliary summary.

        Parameters
        ----------
        features : Tensor
            The input rows.

        Returns
        -------
        tuple[Tensor, Tensor]
            The hidden rows followed by their feature sums.
        """
        hidden = self.linear(features)

        return (hidden, hidden.sum(dim=-1))


class TinyModel(nn.Module):
    """A small model whose hidden layer returns a tensor-bearing tuple."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = _TupleLayer()

    @override
    def forward(self, features: Tensor, refuse: Tensor | None = None) -> Tensor:
        """Return twice the selected hidden output.

        Parameters
        ----------
        features : Tensor
            The input rows.
        refuse : Tensor | None
            A signal that makes this demonstration pass fail after reaching the site,
            or None for a complete pass.

        Returns
        -------
        Tensor
            The model output.

        Raises
        ------
        RuntimeError
            If a refusal signal is supplied.
        """
        pair = self.hidden(features)

        if refuse is not None:
            message = "the tiny model refused this input"
            raise RuntimeError(message)

        return pair[0] * 2.0


class ExampleRun(NamedTuple):
    """Values produced by the external-model instrumentation example.

    Parameters
    ----------
    given : Tensor
        The input passed to the model.
    baseline : Tensor
        The uninstrumented model output.
    observed : Tensor
        The selected hidden value recorded by the observation region.
    observed_axes : tuple[str, ...]
        The axis names recorded with that value.
    patched : Tensor
        The model output while a zero patch is installed.
    after_patch : Tensor
        The output after the successful patching region closes.
    failure : str
        The failure raised after a patched site was reached.
    after_failure : Tensor
        The output after the failing patching region closes.
    unknown_site_refusal : str
        The refusal raised for a site the declaration does not publish.
    after_refusal : Tensor
        The output after the refused region.
    """

    given: Tensor
    baseline: Tensor
    observed: Tensor
    observed_axes: tuple[str, ...]
    patched: Tensor
    after_patch: Tensor
    failure: str
    after_failure: Tensor
    unknown_site_refusal: str
    after_refusal: Tensor


MODEL: Final = ComponentId("tiny-third-party")
HIDDEN: Final = Site(MODEL, Role.RESIDUAL_STREAM, (0,))


def _as_tensor(output: Output) -> Tensor:
    """Return a model output known to be a tensor.

    Parameters
    ----------
    output : Output
        The output to narrow.

    Returns
    -------
    Tensor
        The tensor output.

    Raises
    ------
    TypeError
        If the model returned a tuple.
    """
    if not isinstance(output, torch.Tensor):
        message = f"the tiny model returned a tuple instead of a tensor: {output!r}"
        raise TypeError(message)

    return output


def _declared_model(model: TinyModel) -> ExternalModel:
    """Return the model wrapped through its lowered site declaration.

    Parameters
    ----------
    model : TinyModel
        The model to wrap.

    Returns
    -------
    ExternalModel
        The model and the runtime map lowered from its declaration.
    """
    declaration = ExternalModelSpec(
        model=str(MODEL),
        sites=(
            ExternalSiteDecl(
                path="hidden",
                role="RESIDUAL_STREAM",
                layer=0,
                axes=("batch", "feature"),
                dtype_class="real",
                differentiable=True,
                access=("OBSERVE", "PATCH"),
                select=0,
            ),
        ),
    )

    return ExternalModel(model, ExternalSiteMapIso().forward(declaration))


def _failing_patch(external: ExternalModel, given: Tensor) -> str:
    """Return the failure raised after a patched site is reached.

    Parameters
    ----------
    external : ExternalModel
        The wrapped model.
    given : Tensor
        The ordinary model input.

    Returns
    -------
    str
        The failure message.

    Raises
    ------
    RuntimeError
        If the model unexpectedly completes the pass.
    """
    try:
        with instrument(external, patch={HIDDEN: Zero()}) as region:
            region(lambda: external.run({"features": given, "refuse": torch.ones(())}))

    except RuntimeError as error:
        return str(error)

    message = "the refusal demonstration unexpectedly completed"
    raise RuntimeError(message)


def _unknown_site_refusal(external: ExternalModel) -> str:
    """Return the refusal for a site absent from the declaration.

    Parameters
    ----------
    external : ExternalModel
        The wrapped model.

    Returns
    -------
    str
        The refusal message.

    Raises
    ------
    RuntimeError
        If the undeclared site is unexpectedly accepted.
    """
    unknown = Site(MODEL, Role.GATE, (0,))

    try:
        with instrument(external, observe=frozenset({unknown})):
            pass

    except ValueError as error:
        return str(error)

    message = "the undeclared-site demonstration unexpectedly opened"
    raise RuntimeError(message)


def run_example() -> ExampleRun:
    """Run observation, patching, restoration, and refusal demonstrations.

    Returns
    -------
    ExampleRun
        The values and named refusals produced by the demonstrations.
    """
    external = _declared_model(TinyModel())
    given = torch.tensor([[1.0, 2.0]])
    baseline = _as_tensor(external.run({"features": given}))

    with instrument(external, observe=frozenset({HIDDEN})) as observed_region:
        observed_region(lambda: external.run({"features": given}))

    observed = observed_region.trace[HIDDEN]

    with instrument(external, patch={HIDDEN: Zero()}) as patched_region:
        patched = _as_tensor(patched_region(lambda: external.run({"features": given})))

    after_patch = _as_tensor(external.run({"features": given}))
    failure = _failing_patch(external, given)
    after_failure = _as_tensor(external.run({"features": given}))
    unknown_site_refusal = _unknown_site_refusal(external)
    after_refusal = _as_tensor(external.run({"features": given}))

    return ExampleRun(
        given,
        baseline,
        observed.tensor,
        observed.axes,
        patched,
        after_patch,
        failure,
        after_failure,
        unknown_site_refusal,
        after_refusal,
    )


def main() -> None:
    """Print the results of the external-model instrumentation example."""
    result = run_example()
    lines = (
        f"observed hidden value: {result.observed.tolist()}",
        f"patched model output: {result.patched.tolist()}",
        (
            "restored after successful patch: "
            f"{torch.equal(result.baseline, result.after_patch)}"
        ),
        f"failing pass: {result.failure}",
        (
            "restored after failing patch: "
            f"{torch.equal(result.baseline, result.after_failure)}"
        ),
        f"unknown site: {result.unknown_site_refusal}",
    )
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
