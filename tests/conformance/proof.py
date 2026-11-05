"""Reusable proof-search port conformance checks."""

from typing import Final

import pytest

from lofi.judge.laws import (
    ProofCalculusFixture,
    ProofSchemaFixture,
    StructuralErasureFixture,
    application_and_abstraction_channels_are_exact,
    erasure_is_core_stable,
    erasure_matches_structural_cases,
    erasure_preserves_declared_lineages,
    hypotheses_are_stable_and_injective,
    proof_constructors_return_owned_terms,
    schemata_are_stable,
    schemata_are_structurally_deterministic,
    schemata_match_declared_orientations,
    variable_channels_are_canonical,
)

__all__ = [
    "PROOF_CALCULUS_CONFORMANCE_LAWS",
    "PROOF_SCHEMA_CONFORMANCE_LAWS",
    "STRUCTURAL_ERASURE_CONFORMANCE_LAWS",
    "ProofCalculusConformance",
    "ProofSchemaConformance",
    "StructuralErasureConformance",
]


class ProofSchemaConformance:
    """Conformance laws for oriented proof schemata."""

    pytestmark = pytest.mark.conformance

    def test_schemata_match_declared_orientations(
        self,
        proof_schema_fixture: ProofSchemaFixture,
    ) -> None:
        """Run the declared-orientation law.

        Parameters
        ----------
        proof_schema_fixture : ProofSchemaFixture
        """
        schemata_match_declared_orientations(proof_schema_fixture)

    def test_schemata_are_stable(
        self,
        proof_schema_fixture: ProofSchemaFixture,
    ) -> None:
        """Run the schema-stability law.

        Parameters
        ----------
        proof_schema_fixture : ProofSchemaFixture
        """
        schemata_are_stable(proof_schema_fixture)

    def test_schemata_are_structurally_deterministic(
        self,
        proof_schema_fixture: ProofSchemaFixture,
    ) -> None:
        """Run the structural-determinism law.

        Parameters
        ----------
        proof_schema_fixture : ProofSchemaFixture
        """
        schemata_are_structurally_deterministic(proof_schema_fixture)


PROOF_SCHEMA_CONFORMANCE_LAWS: Final = (
    ProofSchemaConformance.test_schemata_match_declared_orientations,
    ProofSchemaConformance.test_schemata_are_stable,
    ProofSchemaConformance.test_schemata_are_structurally_deterministic,
)


class StructuralErasureConformance:
    """Conformance laws for structural erasures."""

    pytestmark = pytest.mark.conformance

    def test_erasure_preserves_declared_lineages(
        self,
        structural_erasure_fixture: StructuralErasureFixture,
    ) -> None:
        """Run the lineage law.

        Parameters
        ----------
        structural_erasure_fixture : StructuralErasureFixture
        """
        erasure_preserves_declared_lineages(structural_erasure_fixture)

    def test_erasure_matches_structural_cases(
        self,
        structural_erasure_fixture: StructuralErasureFixture,
    ) -> None:
        """Run the structural-case law.

        Parameters
        ----------
        structural_erasure_fixture : StructuralErasureFixture
        """
        erasure_matches_structural_cases(structural_erasure_fixture)

    def test_erasure_is_core_stable(
        self,
        structural_erasure_fixture: StructuralErasureFixture,
    ) -> None:
        """Run the core-stability law.

        Parameters
        ----------
        structural_erasure_fixture : StructuralErasureFixture
        """
        erasure_is_core_stable(structural_erasure_fixture)


STRUCTURAL_ERASURE_CONFORMANCE_LAWS: Final = (
    StructuralErasureConformance.test_erasure_preserves_declared_lineages,
    StructuralErasureConformance.test_erasure_matches_structural_cases,
    StructuralErasureConformance.test_erasure_is_core_stable,
)


class ProofCalculusConformance:
    """Conformance laws for proof calculi."""

    pytestmark = pytest.mark.conformance

    def test_hypotheses_are_stable_and_injective(
        self,
        proof_calculus_fixture: ProofCalculusFixture,
    ) -> None:
        """Run the hypothesis law.

        Parameters
        ----------
        proof_calculus_fixture : ProofCalculusFixture
        """
        hypotheses_are_stable_and_injective(proof_calculus_fixture)

    def test_variable_channels_are_canonical(
        self,
        proof_calculus_fixture: ProofCalculusFixture,
    ) -> None:
        """Run the variable-channel law.

        Parameters
        ----------
        proof_calculus_fixture : ProofCalculusFixture
        """
        variable_channels_are_canonical(proof_calculus_fixture)

    def test_application_and_abstraction_channels_are_exact(
        self,
        proof_calculus_fixture: ProofCalculusFixture,
    ) -> None:
        """Run the compound-constructor law.

        Parameters
        ----------
        proof_calculus_fixture : ProofCalculusFixture
        """
        application_and_abstraction_channels_are_exact(proof_calculus_fixture)

    def test_proof_constructors_return_owned_terms(
        self,
        proof_calculus_fixture: ProofCalculusFixture,
    ) -> None:
        """Run the constructor-ownership law.

        Parameters
        ----------
        proof_calculus_fixture : ProofCalculusFixture
        """
        proof_constructors_return_owned_terms(proof_calculus_fixture)


PROOF_CALCULUS_CONFORMANCE_LAWS: Final = (
    ProofCalculusConformance.test_hypotheses_are_stable_and_injective,
    ProofCalculusConformance.test_variable_channels_are_canonical,
    ProofCalculusConformance.test_application_and_abstraction_channels_are_exact,
    ProofCalculusConformance.test_proof_constructors_return_owned_terms,
)
