import json
import os
import subprocess
import sys
import tomllib
from importlib.metadata import EntryPoints, entry_points
from pathlib import Path
from textwrap import dedent

import pytest

from lofi.registry import Registry, TagCollisionError
from lofi.registry import _registry as registry_module
from lofi.spec import RESOLVER_GROUP
from lofi.spec import _plugins as plugin_module

ROOT = Path(__file__).resolve().parents[3]


class PluginPoint:
    """Test declaration entry point."""

    def __init__(
        self,
        name: str,
        value: str,
        group: str,
        loaded: list[str],
    ) -> None:
        self.name = name
        self.value = value
        self.group = group
        self.dist = None
        self._loaded = loaded

    def load(self) -> None:
        """Record this entry point's value."""
        self._loaded.append(self.value)


def test_plugins_load_in_stable_name_and_value_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[str] = []
    families = (
        PluginPoint("zeta", "family.zeta", plugin_module.FAMILY_GROUP, loaded),
        PluginPoint("alpha", "family.alpha", plugin_module.FAMILY_GROUP, loaded),
    )
    component_groups = ("lofi.alpha", "lofi.zeta")
    plugin_groups = (
        "lofi.rule",
        "lofi.erasure",
        "lofi.matching",
        "lofi.normal_form",
        "lofi.goal",
        "lofi.interpretation",
        "lofi.parse_strategy",
        "lofi.span_index",
        "lofi.chart_store",
        "lofi.cell_pruner",
        "lofi.semiring",
        "lofi.astar_strategy",
        "lofi.astar_root_mode",
        "lofi.astar_heuristic",
        "lofi.astar_proposer",
        "lofi.mbr_utility",
        "lofi.decode_method",
        "lofi.intervention",
        "lofi.acdc_threshold",
        "lofi.das_search",
        "lofi.formalism",
        "lofi.unsatisfiable_policy",
        "lofi.logical_role",
        "lofi.logical_constant",
        "lofi.inference_response",
        "lofi.inference_labeling",
        "lofi.control_variate",
        "lofi.fuzzy_logic",
        "lofi.structure_source",
        "lofi.knowledge_base",
        "lofi.mixing",
        "lofi.step",
        "lofi.chart_composition",
        "lofi.split_weigher",
        "lofi.span_function",
        "lofi.lf_class",
        "lofi.pair_strategy",
        "lofi.corruption",
        "lofi.ill_typing",
        "lofi.reducer",
        "lofi.description_cost",
        "lofi.correction",
        "lofi.experiment_config",
        "lofi.launcher",
        "lofi.sweep_design",
        *component_groups,
    )
    plugins = tuple(
        PluginPoint(name, value, group, loaded)
        for group in plugin_groups
        for name, value in (
            ("zeta", "second.module"),
            ("beta", "second.module"),
            ("alpha", "first.module"),
        )
    )
    scans = [0]

    def installed() -> tuple[PluginPoint, ...]:
        scans[0] += 1
        return (*families, *plugins)

    monkeypatch.setattr(registry_module, "entry_points", installed)
    names = plugin_module.load_plugins()

    assert names == (
        "alpha",
        "zeta",
        *(("alpha", "beta", "zeta") * len(plugin_groups)),
    )
    assert loaded == [
        "family.alpha",
        "family.zeta",
        *(["first.module", "second.module", "second.module"] * len(plugin_groups)),
    ]
    assert scans == [1]


def test_an_empty_plugin_group_loads_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scans = [0]

    def empty() -> EntryPoints:
        scans[0] += 1
        return EntryPoints()

    monkeypatch.setattr(registry_module, "entry_points", empty)

    assert plugin_module.load_plugins() == ()
    assert scans == [1]


def test_plugin_discovery_is_shared_across_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scans = [0]

    def empty() -> EntryPoints:
        scans[0] += 1
        return EntryPoints()

    monkeypatch.setattr(registry_module, "entry_points", empty)

    assert plugin_module.load_plugins() == ()
    assert plugin_module.load_plugins() == ()
    assert scans == [1]


def test_two_family_roots_cannot_claim_one_family_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[str] = []
    families = (
        PluginPoint("third", "third_a:ThirdSpec", plugin_module.FAMILY_GROUP, loaded),
        PluginPoint("third", "third_b:ThirdSpec", plugin_module.FAMILY_GROUP, loaded),
    )
    monkeypatch.setattr(registry_module, "entry_points", lambda: families)

    with pytest.raises(TagCollisionError, match="each declare 'third'"):
        plugin_module.component_groups()


def test_installed_metadata_declares_each_component_family_once() -> None:
    points = tuple(entry_points(group=plugin_module.FAMILY_GROUP))

    assert len({point.name for point in points}) == len(points)
    assert all(point.load().family() == point.name for point in points)


def test_data_and_output_components_have_one_registration_route() -> None:
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = manifest["project"]["entry-points"]
    families = groups[plugin_module.FAMILY_GROUP]

    assert {"dataset", "evaluator", "sink"} <= set(families)
    assert all(
        f"lofi.{family}" in groups for family in ("dataset", "evaluator", "sink")
    )
    assert "lofi.plugins" not in groups


def test_manifest_does_not_advertise_undiscovered_extension_routes() -> None:
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = manifest["project"]["entry-points"]

    assert "lofi.constructor" not in groups
    assert "lofi.lexicon" not in groups
    assert "lofi.plugins" not in groups


def test_every_declared_entry_point_group_has_a_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = manifest["project"]["entry-points"]
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for source_root in (ROOT / "lofi", ROOT / "tests")
        for path in source_root.rglob("*.py")
        if path != Path(__file__).resolve()
    )
    read = {
        group
        for group in groups
        if any(
            spelling in sources
            for spelling in (
                f'"{group}"',
                f"'{group}'",
                f'"{group.removeprefix("lofi.")}"',
                f"'{group.removeprefix('lofi.')}'",
            )
        )
    }
    read.update({plugin_module.FAMILY_GROUP, RESOLVER_GROUP})
    read.update(f"lofi.{family}" for family in groups[plugin_module.FAMILY_GROUP])

    loaded: list[str] = []

    def empty(registry: Registry[object], /) -> tuple[()]:
        loaded.append(registry.group)
        return ()

    monkeypatch.setattr(Registry, "entries", empty)
    plugin_module.load_plugins()
    read.update(loaded)

    unread = tuple(group for group in groups if group not in read)

    assert unread == ()

    planted_group = "lofi." + "unconsumed_test_group"
    planted = groups | {planted_group: {"unused": "unused.module:value"}}
    planted_unread = tuple(group for group in planted if group not in read)

    assert planted_unread == (planted_group,)


def test_installed_metadata_declares_each_open_declaration_family() -> None:
    expected = {
        "lofi.acquisition": {"rc2.1"},
        "lofi.adaptation": {"full", "lora"},
        "lofi.control_variate": {"moving"},
        "lofi.baseline_system": {"bisk-hockenmaier", "lm-prover"},
        "lofi.batch_sampler": {"length-bucket", "random", "sequential"},
        "lofi.corruption": {"deletion", "insertion", "substitution", "swap"},
        "lofi.ill_typing": {"argument-swap", "over-application", "sort-mismatch"},
        "lofi.inference_response": {"judgment", "label"},
        "lofi.inference_labeling": {"graded", "rte", "three-way"},
        "lofi.judgment_source": {"predicted"},
        "lofi.goal": {"cover", "root"},
        "lofi.gold_aggregation": {"majority"},
        "lofi.interpretation": {"category", "dependency", "meaning", "string"},
        "lofi.parse_strategy": {"cky"},
        "lofi.span_index": {"contiguous"},
        "lofi.chart_store": {"native"},
        "lofi.cell_pruner": {"top-k"},
        "lofi.semiring": {
            "boolean",
            "counting",
            "expectation",
            "forest",
            "k-best",
            "log-inside",
            "viterbi",
        },
        "lofi.astar_strategy": {"binary", "linear"},
        "lofi.astar_root_mode": {"free", "goal"},
        "lofi.astar_heuristic": {"supertag-outside", "zero"},
        "lofi.astar_proposer": {"supertag-beam"},
        "lofi.mbr_utility": {"constituent"},
        "lofi.decode_method": {"mbr", "viterbi"},
        "lofi.intervention": {"mean", "noise", "resample", "shuffle", "zero"},
        "lofi.acdc_threshold": {"absolute", "relative"},
        "lofi.das_search": {"boundless", "fixed-rank"},
        "lofi.matching": {
            "alpha",
            "feature-erasure",
            "identity",
            "modal",
            "product",
            "subsumption",
        },
        "lofi.normal_form": {
            "all-of",
            "eisner",
            "hockenmaier-bisk",
            "none",
            "undirected-eisner",
        },
        "lofi.oov_initialization": {"normal", "zeros"},
        "lofi.pair_strategy": {"independent", "overlap-matched"},
        "lofi.lexicon_generator": {"entailment-graph", "masked-lm"},
        "lofi.pair_scorer": {
            "guided-proof",
            "model-checking",
            "neural-proof",
            "prover",
        },
        "lofi.prediction_format": {"lofi-json"},
        "lofi.reducer": {
            "coherence-collapse",
            "cross-lingual",
            "crossed-recovery",
            "encoder-ablation",
            "node-vocabulary",
            "rank-churn",
            "rescoring",
            "restitutive-coverage",
            "residual-structure",
            "source-comparison",
            "transfer",
            "variance",
        },
        "lofi.rescoring_metric": {"recall-homogeneity"},
        "lofi.formalism": {"ab", "cfg", "tlg", "undirected-ccg"},
        "lofi.dataset": {
            "ccgbank",
            "ccgbank-reskin",
            "chaosnli",
            "expert-ratings",
            "fracas",
            "mnli",
            "pmb-ccg",
            "pmb-drs",
            "snli",
            "synthetic",
            "synthetic-inference",
            "unli",
        },
        "lofi.description_cost": {"log-alphabet", "per-constructor", "uniform"},
        "lofi.entailment_graph_reader": {"edge-list"},
        "lofi.evaluator": {
            "bracketing",
            "category-side",
            "coherence-diagnostics",
            "coherence-square",
            "constructor-diagnostics",
            "decode",
            "equivalence",
            "expert-categories",
            "grammaticality",
            "held-out-likelihood",
            "inference",
            "lexical-diagnostics",
            "lexical-types",
            "logical-form",
            "reachability",
            "structural-diagnostics",
        },
        "lofi.fuzzy_logic": {"goedel", "lukasiewicz", "product"},
        "lofi.knowledge_base": {"wikidata"},
        "lofi.lf_language": {"neo-davidsonian", "propositional"},
        "lofi.lf_class": {
            "connective",
            "constant",
            "identity",
            "predicate",
            "quantifier",
        },
        "lofi.logical_constant": {"atom", "choice", "most", "part-of", "sum"},
        "lofi.chart_composition": {"gated-residual", "mean-field"},
        "lofi.span_function": {
            "bilinear",
            "cp-bilinear",
            "gated-mlp",
            "hrr",
            "mlp",
            "transformer",
        },
        "lofi.smt_engine": {"cvc5", "z3"},
        "lofi.tptp_engine": {"eprover", "leo3", "vampire"},
        "lofi.logical_role": {
            "biconditional",
            "conditional",
            "conjunction",
            "disjunction",
            "equality",
            "exclusive-or",
            "existential",
            "generalized-universal",
            "negation",
            "universal",
        },
        "lofi.structure_source": {"given", "learned", "sampled"},
        "lofi.split_weigher": {"attend", "marginal"},
        "lofi.tokenizer": {"regex"},
    }

    for group, names in expected.items():
        assert {point.name for point in entry_points(group=group)} == names


def test_every_plugin_family_has_one_group_named_after_it() -> None:
    assert plugin_module.component_groups() == (
        "lofi.active_collector",
        "lofi.baseline_system",
        "lofi.batch_sampler",
        "lofi.chart",
        "lofi.constraint_source",
        "lofi.controller",
        "lofi.dataset",
        "lofi.evaluator",
        "lofi.featurizer",
        "lofi.gold_aggregation",
        "lofi.grounding",
        "lofi.loss",
        "lofi.optimizer",
        "lofi.pair_sampler",
        "lofi.pair_scorer",
        "lofi.probe",
        "lofi.prover",
        "lofi.readout",
        "lofi.scheduler",
        "lofi.sink",
        "lofi.span_decoder",
        "lofi.subword_pooler",
        "lofi.term_decoder",
        "lofi.term_encoder",
        "lofi.token_encoder",
        "lofi.tokenizer",
        "lofi.trainer",
        "lofi.type_sampler",
    )
    assert plugin_module.entry_point_group("chart") == "lofi.chart"


def test_a_fresh_process_loads_a_third_party_family_and_resolver_from_a_document(
    tmp_path: Path,
) -> None:
    family = tmp_path / "third_family.py"
    variant = tmp_path / "third_variant.py"
    resolver = tmp_path / "third_resolver.py"
    metadata = tmp_path / "third_extension-1.0.dist-info"
    metadata.mkdir()
    family.write_text(
        dedent(
            """
            import didactic.api as dx

            from lofi.spec.components.base import ComponentSpec


            class ThirdFamilySpec(
                ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
            ):
                @classmethod
                def family(cls) -> str:
                    return "third_family"
            """
        ),
        encoding="utf-8",
        newline="\n",
    )
    variant.write_text(
        dedent(
            """
            from typing import Literal, NamedTuple

            from lofi.registry import ComponentPlugin
            from third_family import ThirdFamilySpec


            class ThirdSettings(NamedTuple):
                label: str


            class ThirdSpec(ThirdFamilySpec, extra="forbid"):
                kind: Literal["third"] = "third"
                label: str

                def settings(self) -> ThirdSettings:
                    return ThirdSettings(self.label)


            def _build(settings: ThirdSettings, context: object, /) -> str:
                del context
                return settings.label


            PLUGIN = ComponentPlugin(spec=ThirdSpec, build=_build)
            """
        ),
        encoding="utf-8",
        newline="\n",
    )
    resolver.write_text(
        dedent(
            """
            def upper(*args: str) -> str:
                if len(args) != 1:
                    raise ValueError("third.upper takes one argument")
                return args[0].upper()
            """
        ),
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: third-extension\nVersion: 1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        dedent(
            """
            [lofi.families]
            third_family = third_family:ThirdFamilySpec

            [lofi.third_family]
            third = third_variant:PLUGIN

            [lofi.resolver]
            third.upper = third_resolver:upper
            """
        ),
        encoding="utf-8",
        newline="\n",
    )
    document = tmp_path / "document.json"
    document.write_text(
        json.dumps(
            {
                "component": {"kind": "third", "label": "loaded"},
                "answer": "${third.upper:yes}",
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    script = dedent(
        f"""
        from pathlib import Path

        import didactic.api as dx

        from lofi.spec import SpecLoader, load_plugins

        load_plugins()
        from third_family import ThirdFamilySpec


        class Document(dx.Model, extra="forbid"):
            component: ThirdFamilySpec
            answer: str


        loader = SpecLoader()
        declared = loader.compose_traced(Path({str(document)!r}), schema=Document).spec
        assert type(declared.component).__name__ == "ThirdSpec"
        assert declared.component.label == "loaded"
        assert declared.answer == "YES"
        """
    )
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(ROOT)))
    }

    subprocess.run(
        (sys.executable, "-c", script),
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        encoding="utf-8",
        timeout=30.0,
    )


def test_a_fresh_process_discovers_a_third_party_prover_document(
    tmp_path: Path,
) -> None:
    package = tmp_path / "third_prover.py"
    metadata = tmp_path / "third_prover-1.0.dist-info"
    metadata.mkdir()
    package.write_text(
        dedent(
            """
            from typing import Literal, NamedTuple

            from lofi.registry import ComponentPlugin
            from lofi.spec.provers import ProverSpec


            class ThirdSettings(NamedTuple):
                pass


            class ThirdSpec(ProverSpec, extra="forbid"):
                kind: Literal["third"] = "third"

                def settings(self) -> ThirdSettings:
                    return ThirdSettings()


            def _build(settings: ThirdSettings, context: object, /) -> object:
                return context


            PLUGIN = ComponentPlugin(spec=ThirdSpec, build=_build)
            """
        ),
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: third-prover\nVersion: 1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        "[lofi.prover]\nthird = third_prover:PLUGIN\n", encoding="utf-8", newline="\n"
    )
    root = Path(__file__).parents[3]
    script = dedent(
        """
        from lofi.spec._plugins import load_plugins
        from lofi.spec.provers import ProverSpec

        load_plugins()
        declared = ProverSpec.model_validate({"kind": "third"})
        assert type(declared).__name__ == "ThirdSpec"
        """
    )
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(root)))
    }

    subprocess.run(
        (sys.executable, "-c", script),
        check=True,
        cwd=root,
        env=environment,
        capture_output=True,
        encoding="utf-8",
        timeout=30.0,
    )


def test_a_fresh_process_discovers_and_combines_a_third_party_normal_form(
    tmp_path: Path,
) -> None:
    package = tmp_path / "third_normal_form.py"
    metadata = tmp_path / "third_normal_form-1.0.dist-info"
    metadata.mkdir()
    package.write_text(
        dedent(
            """
            from typing import Literal

            from lofi.judge import NoNormalForm
            from lofi.spec.grammar import NormalFormPlugin, NormalFormSpec


            class ThirdSpec(NormalFormSpec, extra="forbid"):
                kind: Literal["third"] = "third"


            def _build(declaration: ThirdSpec):
                del declaration
                return NoNormalForm()


            PLUGIN = NormalFormPlugin(spec=ThirdSpec, build=_build)
            """
        ),
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: third-normal-form\nVersion: 1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        "[lofi.normal_form]\nthird = third_normal_form:PLUGIN\n",
        encoding="utf-8",
        newline="\n",
    )
    root = Path(__file__).parents[3]
    script = dedent(
        """
        from lofi.judge import AllOf
        from lofi.spec._plugins import load_plugins
        from lofi.spec.grammar import AllOfNormalFormSpec, NormalFormSpec

        load_plugins()
        declared = NormalFormSpec.model_validate({"kind": "third"})
        combined = AllOfNormalFormSpec.model_validate(
            {
                "kind": "all-of",
                "normal_forms": ({"kind": "third"}, {"kind": "eisner"}),
            }
        ).build()
        assert type(declared).__name__ == "ThirdSpec"
        assert isinstance(combined, AllOf)
        assert tuple(one.name for one in combined.normal_forms) == ("none", "eisner")
        """
    )
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(root)))
    }

    subprocess.run(
        (sys.executable, "-c", script),
        check=True,
        cwd=root,
        env=environment,
        capture_output=True,
        encoding="utf-8",
        timeout=30.0,
    )


def test_a_family_name_must_be_a_nonempty_string() -> None:
    with pytest.raises(ValueError, match="family"):
        plugin_module.entry_point_group("")


def test_discovering_the_plugins_needs_no_optional_corpus_reader() -> None:
    script = dedent(
        """
        import sys
        from importlib.abc import MetaPathFinder

        class Refuse(MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.partition(".")[0] == "lairs":
                    raise ModuleNotFoundError(name)
                return None

        sys.meta_path.insert(0, Refuse())

        from lofi.registry import Registry
        from lofi.spec._plugins import load_plugins

        load_plugins()
        for name in ("ccgbank", "ccgbank-reskin", "fracas", "pmb-ccg", "snli"):
            Registry[object]("dataset").get(name)
        """
    )

    completed = subprocess.run(
        (sys.executable, "-c", script),
        check=False,
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        timeout=60.0,
    )

    assert completed.returncode == 0, completed.stderr
