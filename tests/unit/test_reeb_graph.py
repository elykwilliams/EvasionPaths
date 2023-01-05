import unittest


class TestReebGraph(unittest.TestCase):

    def test_determine_case(self):
        hist1 = {"A", "B", "C", "D"}
        hist2 = {"A", "B", "D"}
        case = (hist2.difference(hist1), hist1.difference(hist2))
        assert case == (0, 1)

if __name__ == '__main__':
    unittest.main()
