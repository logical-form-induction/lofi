from lofi._alignment import Alignment as SharedAlignment
from lofi.data import Alignment as DataAlignment


def test_data_and_neural_components_share_one_alignment_type() -> None:
    data_alignment = DataAlignment(((0, 1), None))
    neural_alignment = SharedAlignment(((0, 1), None))

    assert DataAlignment is SharedAlignment
    assert type(data_alignment) is SharedAlignment
    assert type(neural_alignment) is SharedAlignment
    assert data_alignment == neural_alignment
