from utilities import SetDifference


class TestSetDifference:
    def test_init(self):
        L1 = [1, 2, 3]
        L2 = [2, 4, 4]
        sd = SetDifference(L1, L2)
        assert sd.new_list == L1 and sd.old_list == L2

    def test_added(self):
        L1 = [1, 2, 3]
        L2 = [2, 4, 4]
        sd = SetDifference(L1, L2)
        assert sd.added() == {1, 3}

    def test_removed(self):
        L1 = [1, 2, 3]
        L2 = [2, 4, 4]
        sd = SetDifference(L1, L2)
        assert sd.removed() == {4}
