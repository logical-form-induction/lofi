from lofi.data.corpora import pmb
from lofi.data.corpora.pmb import _codec, _drs


def test_package_re_exports_the_pmb_codec_surfaces() -> None:
    modules = (_codec, _drs)
    expected = {name for module in modules for name in module.__all__}

    assert set(pmb.__all__) == expected

    for module in modules:
        for name in module.__all__:
            assert vars(pmb)[name] is vars(module)[name]


def test_re_exported_codec_reads_one_derivation() -> None:
    source = (
        ":- op(601, xfx, (/)).\n"
        ":- op(601, xfx, (\\)).\n"
        "ccg(1,t(np,dog,[lemma:dog,pos:'NN',sem:'CON',wordnet:'dog.n.01',"
        "verbnet:[],from:0,to:3])).\n"
    )
    codec = pmb.PmbCcgCodec()
    fragment = codec.decode(source)

    assert codec.name == pmb.NAME
    assert len(fragment.records) == 3
    assert codec.decode(codec.encode(fragment.records)) == fragment
