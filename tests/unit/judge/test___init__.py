from lofi import judge
from lofi.judge import (
    _boundary,
    _composition,
    _coordination,
    _errors,
    _index,
    _normal,
    _pattern,
    _ports,
    _proof,
    _raising,
    _recover,
    _ruleset,
    _spine,
    _substitution,
    _values,
)


def test_the_package_re_exports_the_ports_surface_and_recovery() -> None:
    assert set(judge.__all__) >= {*_ports.__all__, *_recover.__all__}

    for name in _ports.__all__:
        assert getattr(judge, name) is getattr(_ports, name)

    assert judge.recover is _recover.recover


def test_the_package_re_exports_the_boundary_surface() -> None:
    for name in _boundary.__all__:
        assert getattr(judge, name) is getattr(_boundary, name)


def test_the_package_re_exports_rule_set_errors() -> None:
    for name in _errors.__all__:
        assert getattr(judge, name) is getattr(_errors, name)


def test_the_package_re_exports_rule_indexing() -> None:
    for name in _index.__all__:
        assert getattr(judge, name) is getattr(_index, name)


def test_the_package_re_exports_normal_forms() -> None:
    for name in _normal.__all__:
        assert getattr(judge, name) is getattr(_normal, name)


def test_the_package_re_exports_the_rule_set_and_its_records() -> None:
    assert set(judge.__all__) >= {*_ruleset.__all__, *_values.__all__}

    for name in (*_ruleset.__all__, *_values.__all__):
        assert getattr(judge, name) is getattr(
            _ruleset if name in _ruleset.__all__ else _values, name
        )


def test_the_package_re_exports_spine_matching() -> None:
    for name in _spine.__all__:
        assert getattr(judge, name) is getattr(_spine, name)


def test_the_package_re_exports_pattern_rules() -> None:
    for name in _pattern.__all__:
        assert getattr(judge, name) is getattr(_pattern, name)


def test_the_package_re_exports_proof_search() -> None:
    for name in _proof.__all__:
        assert getattr(judge, name) is getattr(_proof, name)


def test_the_package_re_exports_generalized_composition() -> None:
    for name in _composition.__all__:
        assert getattr(judge, name) is getattr(_composition, name)


def test_the_package_re_exports_coordination_rules() -> None:
    for name in _coordination.__all__:
        assert getattr(judge, name) is getattr(_coordination, name)


def test_the_package_re_exports_generalized_substitution() -> None:
    for name in _substitution.__all__:
        assert getattr(judge, name) is getattr(_substitution, name)


def test_the_package_re_exports_type_raising() -> None:
    for name in _raising.__all__:
        assert getattr(judge, name) is getattr(_raising, name)


def test_recovery_runs_through_the_package_namespace() -> None:
    assert judge.recover((), [0], 0) == ()
