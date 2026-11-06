import math
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from torch import nn

from lofi.nn._values import Encoded
from lofi.nn.spacemap import MapForm, MultilayerOptions, TypeSpaceMap

if TYPE_CHECKING:
    from torch import Tensor

SOURCE: Final = 4
TARGET: Final = 6
TOLERANCE: Final = 1e-5


def build(
    form: MapForm = MapForm.LINEAR,
    *,
    source: int = SOURCE,
    target: int = TARGET,
    layers: MultilayerOptions | None = None,
) -> TypeSpaceMap:
    torch.manual_seed(19)

    if form is MapForm.MULTILAYER and layers is None:
        layers = MultilayerOptions(hidden=64, depth=2, dropout=0.1)

    return TypeSpaceMap(
        source_width=source, target_width=target, form=form, layers=layers
    )


def basis(source: int) -> Encoded:
    return Encoded(torch.eye(source), ("row", "width"))


def matrix(subject: TypeSpaceMap) -> Tensor:
    """Return the map of the standard basis: one row per source dimension."""
    return subject.apply_to(basis(subject.source_width)).tensor


def fill(subject: TypeSpaceMap, values: Tensor) -> None:
    with torch.no_grad():
        for parameter in subject.parameters():
            parameter.copy_(values[: parameter.numel()].reshape(parameter.shape))


def test_the_form_is_selected_by_the_declared_field() -> None:
    assert build(MapForm.LINEAR).form is MapForm.LINEAR
    assert build(MapForm.MULTILAYER).form is MapForm.MULTILAYER
    assert build(MapForm.LINEAR).layers is None
    assert build(MapForm.MULTILAYER).layers == MultilayerOptions(
        hidden=64, depth=2, dropout=0.1
    )


def test_the_map_reports_its_widths() -> None:
    subject = build()

    assert subject.source_width == SOURCE
    assert subject.target_width == TARGET
    assert "linear, 4 -> 6" in repr(subject)
    assert "multilayer, 4 -> 6" in repr(build(MapForm.MULTILAYER))


def test_a_form_that_is_not_one_is_refused() -> None:
    with pytest.raises(TypeError, match="MapForm"):
        TypeSpaceMap(
            source_width=SOURCE,
            target_width=TARGET,
            form=cast("MapForm", "linear"),
            layers=None,
        )


@pytest.mark.parametrize(("source", "target"), [(0, TARGET), (SOURCE, 0)])
def test_a_width_that_is_not_positive_is_refused(source: int, target: int) -> None:
    with pytest.raises(ValueError, match="zero"):
        TypeSpaceMap(
            source_width=source,
            target_width=target,
            form=MapForm.LINEAR,
            layers=None,
        )


def test_the_linear_form_takes_no_layers() -> None:
    with pytest.raises(ValueError, match="takes no layers"):
        build(
            MapForm.LINEAR,
            layers=MultilayerOptions(hidden=64, depth=2, dropout=0.1),
        )


def test_the_linear_form_is_one_linear_layer() -> None:
    subject = build()
    modules = [module for module in subject.modules() if module is not subject]

    assert [type(module) for module in modules] == [nn.Linear]
    assert dict(subject.state_dict()).keys() == {"_map.weight", "_map.bias"}
    assert tuple(subject.state_dict()["_map.weight"].shape) == (TARGET, SOURCE)


def test_the_linear_form_is_affine_in_its_weight_and_bias() -> None:
    subject = build()
    saved = subject.state_dict()
    generator = torch.Generator().manual_seed(5)
    vectors = torch.randn(7, SOURCE, generator=generator)
    mapped = subject.apply_to(Encoded(vectors, ("row", "width"))).tensor

    assert tuple(mapped.shape) == (7, TARGET)
    assert torch.allclose(
        mapped, vectors @ saved["_map.weight"].T + saved["_map.bias"], atol=TOLERANCE
    )


@pytest.mark.parametrize("form", list(MapForm))
def test_the_gradient_reaches_every_parameter(form: MapForm) -> None:
    subject = build(form)
    subject.eval()
    generator = torch.Generator().manual_seed(5)
    vectors = torch.randn(7, SOURCE, generator=generator)

    subject.apply_to(
        Encoded(vectors, ("row", "width"))
    ).tensor.square().sum().backward()

    for name, parameter in subject.named_parameters():
        assert parameter.grad is not None, name
        assert bool(parameter.grad.abs().sum() > 0), name


def test_the_multilayer_form_has_the_declared_layers() -> None:
    options = MultilayerOptions(hidden=5, depth=3, dropout=0.25)
    subject = build(MapForm.MULTILAYER, layers=options)
    linears = [module for module in subject.modules() if type(module) is nn.Linear]
    dropouts = [module for module in subject.modules() if type(module) is nn.Dropout]

    assert subject.layers == options
    assert [type(module) for module in subject.modules() if module is not subject] == [
        nn.Sequential,
        nn.Linear,
        nn.Tanh,
        nn.Dropout,
        nn.Linear,
        nn.Tanh,
        nn.Dropout,
        nn.Linear,
    ]
    assert [(linear.in_features, linear.out_features) for linear in linears] == [
        (SOURCE, 5),
        (5, 5),
        (5, TARGET),
    ]
    assert [dropout.p for dropout in dropouts] == [0.25, 0.25]


def test_a_multilayer_form_of_depth_one_is_one_linear_layer() -> None:
    subject = build(
        MapForm.MULTILAYER,
        layers=MultilayerOptions(hidden=64, depth=1, dropout=0.1),
    )
    modules = [module for module in subject.modules() if module is not subject]

    assert [type(module) for module in modules] == [nn.Sequential, nn.Linear]


def test_the_multilayer_form_drops_units_in_training_only() -> None:
    subject = build(
        MapForm.MULTILAYER,
        layers=MultilayerOptions(hidden=64, depth=2, dropout=0.5),
    )
    vectors = Encoded(torch.ones(8, SOURCE), ("row", "width"))

    subject.train()
    torch.manual_seed(1)
    first = subject.apply_to(vectors).tensor
    torch.manual_seed(2)
    second = subject.apply_to(vectors).tensor

    assert not torch.equal(first, second)

    subject.eval()

    assert torch.equal(
        subject.apply_to(vectors).tensor, subject.apply_to(vectors).tensor
    )


def test_the_multilayer_form_is_not_linear() -> None:
    subject = build(MapForm.MULTILAYER)
    subject.eval()
    generator = torch.Generator().manual_seed(5)
    vectors = torch.randn(7, SOURCE, generator=generator)
    mapped = subject.apply_to(Encoded(vectors, ("row", "width"))).tensor
    lifted = subject.apply_to(Encoded(2.0 * vectors, ("row", "width"))).tensor
    origin = subject.apply_to(Encoded(torch.zeros(1, SOURCE), ("row", "width"))).tensor

    assert tuple(mapped.shape) == (7, TARGET)
    assert not torch.allclose(lifted - origin, 2.0 * (mapped - origin), atol=1e-2)


@pytest.mark.parametrize("form", list(MapForm))
def test_the_map_keeps_the_leading_axes_and_the_mask(form: MapForm) -> None:
    subject = build(form)
    subject.eval()
    mask = torch.tensor([[True, False], [True, True], [False, False]])
    vectors = Encoded(torch.ones(3, 2, SOURCE), ("item", "slot", "width"), mask)
    mapped = subject.apply_to(vectors)

    assert mapped.axes == ("item", "slot", "width")
    assert tuple(mapped.tensor.shape) == (3, 2, TARGET)
    assert mapped.mask is mask


def test_vectors_over_another_innermost_axis_are_refused() -> None:
    with pytest.raises(ValueError, match="innermost"):
        build().apply_to(Encoded(torch.ones(SOURCE, 2), ("width", "row")))


def test_vectors_of_another_width_are_refused() -> None:
    with pytest.raises(ValueError, match="wide"):
        build().apply_to(Encoded(torch.ones(2, SOURCE + 1), ("row", "width")))


@pytest.mark.parametrize("form", list(MapForm))
def test_the_state_reloads_into_a_fresh_map(form: MapForm) -> None:
    subject = build(form)
    fresh = TypeSpaceMap(
        source_width=SOURCE, target_width=TARGET, form=form, layers=subject.layers
    )
    count = sum(parameter.numel() for parameter in subject.parameters())

    fill(subject, torch.linspace(-1.0, 1.0, count))
    fresh.load_state_dict(subject.state_dict())
    subject.eval()
    fresh.eval()

    assert torch.equal(matrix(subject), matrix(fresh))


def test_the_options_keep_the_declared_values() -> None:
    options = MultilayerOptions(hidden=64, depth=2, dropout=0.1)

    assert options.hidden == 64
    assert options.depth == 2
    assert options.dropout == 0.1
    assert options == MultilayerOptions(hidden=64, depth=2, dropout=0.1)
    assert options != MultilayerOptions(hidden=64, depth=3, dropout=0.1)
    assert hash(options) == hash(MultilayerOptions(hidden=64, depth=2, dropout=0.1))
    assert repr(options) == "MultilayerOptions(hidden=64, depth=2, dropout=0.1)"
    assert options.__eq__(object()) is NotImplemented


def test_a_hidden_width_that_is_not_positive_is_refused() -> None:
    with pytest.raises(ValueError, match="zero"):
        MultilayerOptions(hidden=0, depth=2, dropout=0.1)


def test_a_depth_that_is_not_positive_is_refused() -> None:
    with pytest.raises(ValueError, match="zero"):
        MultilayerOptions(hidden=64, depth=0, dropout=0.1)


@pytest.mark.parametrize("rate", [1.0, -0.1, 1.5])
def test_a_dropout_rate_outside_its_range_is_refused(rate: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        MultilayerOptions(hidden=64, depth=2, dropout=rate)


def test_a_dropout_rate_that_is_not_finite_is_refused() -> None:
    with pytest.raises(ValueError, match="finite"):
        MultilayerOptions(hidden=64, depth=2, dropout=math.nan)
