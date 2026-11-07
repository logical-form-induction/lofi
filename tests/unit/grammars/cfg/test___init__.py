from lofi.grammars import cfg


def test_the_package_adds_no_public_python_declaration_route() -> None:
    assert not tuple(name for name in vars(cfg) if not name.startswith("_"))
