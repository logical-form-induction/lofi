from lofi.spec._pattern_laws import provide_rule_laws


def test_provider_declares_both_pattern_cases() -> None:
    cases = provide_rule_laws()

    assert cases.provider == "pattern"
    assert tuple(case.name for case in cases.ordinary) == (
        "pattern/pairing",
        "pattern/unary",
    )
    assert cases.totality == ()
