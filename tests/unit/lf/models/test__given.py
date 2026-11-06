from tests.language import language

from lofi.lf.models import (
    CRISP,
    EXTREME,
    PRODUCT,
    RESIDUUM,
    Frame,
    GivenStructureSource,
    Structure,
    TruthAlgebra,
)


def algebra() -> TruthAlgebra:
    return PRODUCT.build(
        implication=RESIDUUM,
        quantifier=EXTREME,
        equality=CRISP,
        margin=0.0,
    )


def test_a_given_source_holds_one_read_only_theory_checked_structure() -> None:
    declared = language(("p",))
    frame = Frame(
        declared.store,
        function_type=declared.function_type,
        truth_type="t",
        carriers={"e": 1, "v": 1},
        full_function_space_cap=4,
        henkin_pool=2,
        seed=0,
    )
    vocabulary = declared.store.signature.vocabulary(declared.term_sort)
    structure = Structure("intended", frame, {vocabulary.id("p"): True})
    source = GivenStructureSource(
        declared,
        structure,
        algebra(),
        theory_floor=1.0,
    )

    assert source.structures() == (structure,)
    assert source.name == "given"
    assert source.evidence_label == "truth in an intended model"
    assert not source.trainable
    assert source.open_step() is None
