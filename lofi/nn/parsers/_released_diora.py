"""The MLP chart distributed in the NAACL 2019 DIORA release."""

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from lofi.deduce import SpanIndex

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["ReleasedDioraMLP", "unit_vectors"]

_STATE_PREFIX = "module."


def _weight(state: Mapping[str, Tensor], name: str, /) -> Tensor:
    """Return a detached copy of one required released tensor."""
    key = f"{_STATE_PREFIX}{name}"

    if key not in state:
        msg = f"the released DIORA checkpoint has no {key!r} tensor"
        raise ValueError(msg)

    return state[key].detach().clone()


def unit_vectors(vector: Tensor, /) -> Tensor:
    """Return vectors normalized along their final dimension.

    A zero vector has no direction and stays zero; every other vector is divided
    by its own norm.
    """
    norm = vector.norm(dim=-1, keepdim=True)
    zero = norm == 0.0

    return torch.where(zero, vector, vector / torch.where(zero, 1.0, norm))


class ReleasedDioraMLP(nn.Module):
    """Run the released MLP DIORA chart from declared ELMo vectors.

    The modules named ``alpha``, ``inside`` and ``outside`` publish the chart's
    log split attention and its scalar inside and outside scores. The forward
    value is the root inside vector.
    """

    def __init__(self, state: Mapping[str, Tensor]) -> None:
        super().__init__()
        self.embedding = nn.Parameter(_weight(state, "embed.mat"))
        self.root = nn.Parameter(_weight(state, "diora.root_vector_out_h"))
        self.inside_score = nn.Parameter(_weight(state, "diora.inside_score_func.mat"))
        self.outside_score = nn.Parameter(
            _weight(state, "diora.outside_score_func.mat")
        )
        self.inside_leaf = nn.Parameter(_weight(state, "diora.inside_compose_func.V"))
        self.inside_first = nn.Parameter(
            _weight(state, "diora.inside_compose_func.W_0")
        )
        self.inside_second = nn.Parameter(
            _weight(state, "diora.inside_compose_func.W_1")
        )
        self.inside_bias = nn.Parameter(_weight(state, "diora.inside_compose_func.B"))
        self.inside_second_bias = nn.Parameter(
            _weight(state, "diora.inside_compose_func.B_1")
        )
        self.outside_first = nn.Parameter(
            _weight(state, "diora.outside_compose_func.W_0")
        )
        self.outside_second = nn.Parameter(
            _weight(state, "diora.outside_compose_func.W_1")
        )
        self.outside_bias = nn.Parameter(_weight(state, "diora.outside_compose_func.B"))
        self.outside_second_bias = nn.Parameter(
            _weight(state, "diora.outside_compose_func.B_1")
        )
        self.alpha = nn.Identity()
        self.inside = nn.Identity()
        self.outside = nn.Identity()
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        """Refuse released tensors that do not describe one MLP DIORA."""
        if self.embedding.ndim != 2 or self.root.ndim != 1:
            msg = "the released DIORA embedding and root have invalid ranks"
            raise ValueError(msg)

        width = int(self.root.shape[0])
        square = (width, width)
        expected = {
            "embedding": (width, int(self.embedding.shape[1])),
            "inside_score": square,
            "outside_score": square,
            "inside_leaf": square,
            "inside_first": (width * 2, width),
            "inside_second": square,
            "inside_bias": (width,),
            "inside_second_bias": (width,),
            "outside_first": (width * 2, width),
            "outside_second": square,
            "outside_bias": (width,),
            "outside_second_bias": (width,),
        }

        for name, shape in expected.items():
            found = tuple(getattr(self, name).shape)

            if found != shape:
                msg = f"the released DIORA {name} has shape {found}, not {shape}"
                raise ValueError(msg)

    @property
    def input_width(self) -> int:
        """Return how many ELMo features each input token carries."""
        return int(self.embedding.shape[1])

    @staticmethod
    def _compose(
        left: Tensor,
        right: Tensor,
        first: Tensor,
        second: Tensor,
        bias: Tensor,
        second_bias: Tensor,
    ) -> Tensor:
        """Return one MLP composition of paired chart vectors."""
        joined = torch.cat((left, right), dim=-1)
        hidden = torch.relu(joined @ first + bias)
        return torch.relu(hidden @ second + second_bias)

    @staticmethod
    def _bilinear(left: Tensor, matrix: Tensor, right: Tensor) -> Tensor:
        """Return one bilinear score per paired chart vector."""
        return ((left @ matrix) * right).sum(dim=-1)

    @override
    def forward(self, embeddings: Tensor) -> Tensor:
        """Return the root vector after publishing all three chart quantities."""
        if embeddings.ndim != 3:
            msg = (
                "released DIORA inputs have item, token and embedding axes, "
                f"not shape {tuple(embeddings.shape)}"
            )
            raise ValueError(msg)

        if int(embeddings.shape[2]) != int(self.embedding.shape[1]):
            msg = (
                f"released DIORA expects {int(self.embedding.shape[1])} embedding "
                f"features, not {int(embeddings.shape[2])}"
            )
            raise ValueError(msg)

        items, length, _ = embeddings.shape
        index = SpanIndex(int(length))
        projected = embeddings @ self.embedding.mT
        leaves = unit_vectors(
            torch.tanh(projected @ self.inside_leaf + self.inside_bias)
        )
        inside_vectors: dict[tuple[int, int], Tensor] = {
            (position, position + 1): leaves[:, position]
            for position in range(int(length))
        }
        inside_scores: dict[tuple[int, int], Tensor] = {
            span: embeddings.new_zeros(int(items)) for span in inside_vectors
        }
        log_alpha = embeddings.new_full(
            (int(items), index.num_spans, index.max_splits), -torch.inf
        )

        for width in range(2, int(length) + 1):
            for start in range(int(length) - width + 1):
                end = start + width
                candidates: list[Tensor] = []
                totals: list[Tensor] = []

                for split in range(start + 1, end):
                    left = inside_vectors[start, split]
                    right = inside_vectors[split, end]
                    candidates.append(
                        self._compose(
                            left,
                            right,
                            self.inside_first,
                            self.inside_second,
                            self.inside_bias,
                            self.inside_second_bias,
                        )
                    )
                    totals.append(
                        self._bilinear(left, self.inside_score, right)
                        + inside_scores[start, split]
                        + inside_scores[split, end]
                    )

                total = torch.stack(totals, dim=1)
                attention = torch.log_softmax(total, dim=1)
                weights = attention.exp()
                vectors = torch.stack(candidates, dim=1)
                inside_vectors[start, end] = unit_vectors(
                    (vectors * weights.unsqueeze(2)).sum(dim=1)
                )
                inside_scores[start, end] = (total * weights).sum(dim=1)
                span = index.span_id(start, end)
                log_alpha[:, span, : width - 1] = attention

        outside_vectors: dict[tuple[int, int], Tensor] = {
            (0, int(length)): unit_vectors(self.root).expand(int(items), -1)
        }
        outside_scores: dict[tuple[int, int], Tensor] = {
            (0, int(length)): embeddings.new_zeros(int(items))
        }

        for width in range(int(length) - 1, 0, -1):
            for start in range(int(length) - width + 1):
                end = start + width
                contexts: list[Tensor] = []
                totals: list[Tensor] = []

                for parent_end in range(end + 1, int(length) + 1):
                    parent = outside_vectors[start, parent_end]
                    sibling = inside_vectors[end, parent_end]
                    contexts.append(
                        self._compose(
                            sibling,
                            parent,
                            self.outside_first,
                            self.outside_second,
                            self.outside_bias,
                            self.outside_second_bias,
                        )
                    )
                    totals.append(
                        self._bilinear(sibling, self.outside_score, parent)
                        + inside_scores[end, parent_end]
                        + outside_scores[start, parent_end]
                    )

                for parent_start in range(start):
                    parent = outside_vectors[parent_start, end]
                    sibling = inside_vectors[parent_start, start]
                    contexts.append(
                        self._compose(
                            sibling,
                            parent,
                            self.outside_first,
                            self.outside_second,
                            self.outside_bias,
                            self.outside_second_bias,
                        )
                    )
                    totals.append(
                        self._bilinear(sibling, self.outside_score, parent)
                        + inside_scores[parent_start, start]
                        + outside_scores[parent_start, end]
                    )

                total = torch.stack(totals, dim=1)
                weights = torch.softmax(total, dim=1)
                vectors = torch.stack(contexts, dim=1)
                outside_vectors[start, end] = unit_vectors(
                    (vectors * weights.unsqueeze(2)).sum(dim=1)
                )
                outside_scores[start, end] = (total * weights).sum(dim=1)

        inside = torch.stack(
            [inside_scores[index.span_of(span)] for span in range(index.num_spans)],
            dim=1,
        )
        outside = torch.stack(
            [outside_scores[index.span_of(span)] for span in range(index.num_spans)],
            dim=1,
        )
        self.alpha(log_alpha)
        self.inside(inside)
        self.outside(outside)

        return inside_vectors[0, int(length)]
