from lofi.judge import COMPOSITION_RESTRICTIONS
from lofi.spec._composition_laws import provide_rule_laws

VIEWS = ("direct", "swapped", "undirected")
ROLES = ("forward", "backward")


def test_provider_declares_the_full_application_and_composition_matrices() -> None:
    cases = provide_rule_laws()
    applications = {f"application/{view}/{role}" for view in VIEWS for role in ROLES}
    compositions = {
        f"composition/{view}/{role}/{order}/{variant}/{direction}"
        for view in VIEWS
        for role in ROLES
        for order in ("2", "any")
        for variant in ("harmonic", "crossed")
        for direction in COMPOSITION_RESTRICTIONS
    }
    totality = len(ROLES) * (1 + 3 * 2 * len(COMPOSITION_RESTRICTIONS))

    assert cases.provider == "composition"
    assert len(cases.ordinary) == len(applications | compositions)
    assert {case.name for case in cases.ordinary} == applications | compositions
    assert len(cases.totality) == totality
    assert len({case.name for case in cases.totality}) == totality


def test_r3_composition_cases_cover_application_every_order_and_the_schema() -> None:
    names = tuple(case.name for case in provide_rule_laws().totality)
    directions = len(COMPOSITION_RESTRICTIONS)

    assert sum(name.startswith("r3/application/") for name in names) == 2
    assert sum("/1/" in name for name in names) == 4 * directions
    assert sum("/2/" in name for name in names) == 4 * directions
    assert sum("/any/" in name for name in names) == 4 * directions


def test_each_case_checks_a_rule_the_grammar_compiler_built() -> None:
    names = {case.rule_name for case in provide_rule_laws().ordinary}

    assert "application:fwd" in names
    assert "composition:bwd:2:crossed:kuhlmann" in names
    assert "composition:arrow:any:harmonic:steedman" in names
