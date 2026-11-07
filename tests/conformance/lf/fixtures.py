"""Canonical logical-form conformance fixtures."""

from typing import TYPE_CHECKING as _TYPE_CHECKING

from tests.conformance.lf.corpus import conformance_language as _language

if _TYPE_CHECKING:
    from lofi.terms import Term as _Term

__all__ = [
    "abs2_et_const",
    "abs_et_const",
    "abs_t_const",
    "appl2_t_const",
    "appl_et_const",
    "appl_et_distinct_vars",
    "appl_t_const",
    "appl_t_distinct_vars",
    "eat",
    "f0_eet",
    "f0_et",
    "jo",
    "nested_abstractions",
    "nested_quantifiers",
    "prop_appl_et_const",
    "quant2_t_const",
    "quant_t_const",
    "sleep",
    "x0_e",
    "x1_e",
]


_LANGUAGE = _language()
_LF_STORE = _LANGUAGE.store


def _apply(
    function: _Term,
    argument: _Term,
) -> _Term:
    """Build a logical-form application."""
    return _LANGUAGE.apply(function, argument)


def _bind(
    operator: str,
    variable: str,
    body: _Term,
) -> _Term:
    """Close a named free variable under a binder."""
    abstraction = _LF_STORE.bind(_LANGUAGE.calculus.lam, variable, body)

    if operator == _LANGUAGE.calculus.lam:
        return abstraction

    return _LANGUAGE.apply(_LANGUAGE.constant(operator), abstraction)


jo = _LANGUAGE.constant("jo")
sleep = _LANGUAGE.constant("sleep")
eat = _LANGUAGE.constant("eat")
x0_e = _LF_STORE.fvar(_LANGUAGE.term_sort, "x0_e")
x1_e = _LF_STORE.fvar(_LANGUAGE.term_sort, "x1_e")
f0_et = _LF_STORE.fvar(_LANGUAGE.term_sort, "f0_et")
f0_eet = _LF_STORE.fvar(_LANGUAGE.term_sort, "f0_eet")

appl_et_const = _apply(eat, x0_e)
appl_t_const = _apply(sleep, x0_e)
appl2_t_const = _apply(appl_et_const, x1_e)
quant_t_const = _bind("forall", "x0_e", appl_t_const)
quant2_t_const = _bind("forall", "x0_e", appl2_t_const)
abs_et_const = _bind("lam", "x0_e", appl2_t_const)
abs2_et_const = _bind("lam", "x1_e", abs_et_const)
abs_t_const = _bind("lam", "x0_e", appl_t_const)
prop_appl_et_const = _apply(abs2_et_const, jo)

appl_et_distinct_vars = _apply(f0_eet, x0_e)
appl_t_distinct_vars = _apply(f0_et, x0_e)

_nested_function = _LF_STORE.fvar(_LANGUAGE.term_sort, "nested_function")
_nested_argument = _LF_STORE.fvar(_LANGUAGE.term_sort, "nested_argument")
nested_abstractions = _bind(
    "lam",
    "nested_function",
    _bind(
        "lam",
        "nested_argument",
        _apply(_nested_function, _nested_argument),
    ),
)

_quantified_outer = _LF_STORE.fvar(_LANGUAGE.term_sort, "quantified_outer")
_quantified_inner = _LF_STORE.fvar(_LANGUAGE.term_sort, "quantified_inner")
_binary_predicate = _LANGUAGE.constant("P")
nested_quantifiers = _bind(
    "forall",
    "quantified_outer",
    _bind(
        "exists",
        "quantified_inner",
        _apply(
            _apply(_binary_predicate, _quantified_outer),
            _quantified_inner,
        ),
    ),
)
