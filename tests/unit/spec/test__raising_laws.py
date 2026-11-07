from lofi.spec._raising_laws import provide_rule_laws


def test_provider_declares_the_full_raising_matrix() -> None:
    cases = provide_rule_laws()
    expected = {
        f"raising/{view}/{role}/{mode}"
        for view in ("direct", "swapped", "undirected")
        for role in ("forward", "backward")
        for mode in ("targeted", "untargeted")
    }

    assert cases.provider == "raising"
    assert len(cases.ordinary) == 12
    assert len(cases.totality) == 1
    assert {case.name for case in cases.ordinary} == expected


def test_r3_raising_case_is_targeted_and_forward() -> None:
    cases = provide_rule_laws()

    assert cases.totality[0].name == "r3/raising/forward/targeted"
