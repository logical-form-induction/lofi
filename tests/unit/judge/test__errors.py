from lofi.judge._errors import RuleSetValidationError
from lofi.judge._values import Diagnostic


def _findings() -> tuple[Diagnostic, ...]:
    return (
        Diagnostic(
            "forward_enumeration_unavailable",
            "forward_enumeration",
            "rule 'left' lacks complete forward enumeration",
            rules=("left",),
        ),
        Diagnostic(
            "forward_enumeration_unavailable",
            "forward_enumeration",
            "rule 'right' lacks complete forward enumeration",
            rules=("right",),
        ),
    )


def test_the_error_exposes_its_rule_set_requirement_and_findings() -> None:
    findings = _findings()
    error = RuleSetValidationError(
        ruleset="application",
        requirement="forward_enumeration",
        diagnostics=findings,
    )

    assert error.ruleset == "application"
    assert error.requirement == "forward_enumeration"
    assert error.diagnostics == findings


def test_the_error_copies_the_diagnostic_sequence() -> None:
    findings = list(_findings())
    error = RuleSetValidationError(
        ruleset="application",
        requirement="forward_enumeration",
        diagnostics=findings,
    )
    findings.clear()

    assert error.diagnostics == _findings()


def test_the_error_message_names_the_inventory_requirement_and_each_failure() -> None:
    error = RuleSetValidationError(
        ruleset="application",
        requirement="forward_enumeration",
        diagnostics=_findings(),
    )
    message = str(error)

    assert "rule set 'application'" in message
    assert "'forward_enumeration'" in message
    assert "rule 'left' lacks complete forward enumeration" in message
    assert "rule 'right' lacks complete forward enumeration" in message
