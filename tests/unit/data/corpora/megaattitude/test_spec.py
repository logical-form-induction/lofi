import didactic.api as dx
import pytest

from lofi.data.corpora.megaattitude import (
    MegaAttitudeRelationSpec,
    MegaAttitudeSpec,
)


def test_source_metadata_must_be_present() -> None:
    with pytest.raises(dx.ValidationError, match="cannot be empty"):
        MegaAttitudeSpec(
            source_id="",
            release_id="v2.1",
            language="en",
            source_method="elicited",
            spdx="CC-BY-SA-4.0",
        )


@pytest.mark.parametrize(
    "relations",
    [
        (),
        (
            MegaAttitudeRelationSpec(response="yes", relation="entails"),
            MegaAttitudeRelationSpec(response="yes", relation="neutral"),
        ),
        (
            MegaAttitudeRelationSpec(response="yes", relation="entails"),
            MegaAttitudeRelationSpec(response="maybe", relation="entails"),
        ),
    ],
)
def test_the_veridicality_relation_table_is_a_nonempty_bijection(
    relations: tuple[MegaAttitudeRelationSpec, ...],
) -> None:
    with pytest.raises(dx.ValidationError, match="nonempty bijection"):
        MegaAttitudeSpec(
            source_id="megaattitude",
            release_id="v2.1",
            language="en",
            source_method="elicited",
            spdx="CC-BY-SA-4.0",
            veridicality_relations=relations,
        )


def test_a_relation_table_field_cannot_be_blank() -> None:
    with pytest.raises(dx.ValidationError, match="cannot be empty"):
        MegaAttitudeRelationSpec(response="", relation="entails")
