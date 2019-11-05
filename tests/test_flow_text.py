import data_algebra.flow_text


def test_flow_text():
    strs = ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg"]
    flowed = data_algebra.flow_text.flow_text(strs, align_right=10, sep_width=2)
    flowed = [", ".join(line) for line in flowed]
    flowed = ",\n ".join(flowed)
    pieces = flowed.split("\n")
    for piece in pieces:
        assert len(piece) <= 12
