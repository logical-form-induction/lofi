import json

import pytest
from didactic.settings import load_document

from lofi.registry import Registry
from lofi.run import section_root
from lofi.spec import LossSection
from lofi.spec._plugins import load_plugins
from lofi.spec.components.loss import LossSpec


def _registered() -> tuple[str, ...]:
    return Registry[object]("loss").available()


@pytest.mark.parametrize("name", _registered())
def test_every_registered_loss_ships_a_preset_that_validates(name: str) -> None:
    load_plugins()
    document = load_document(section_root("loss") / f"{name}.yaml")
    text = json.dumps(document)

    if "terms" in document:
        section = LossSection.model_validate_json(text)
        kinds = {term.objective.kind for term in section.terms}
    else:
        kinds = {LossSpec.model_validate_json(text).kind}

    assert name in kinds
