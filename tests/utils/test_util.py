import unittest

from adept.utils.util import listd_to_dlist, dlist_to_listd


class TestUtil(unittest.TestCase):
    def test_dlist_to_listd(self):
        assert dlist_to_listd({"a": [1]}) == [{"a": 1}]

    def test_listd_to_dlist(self):
        assert listd_to_dlist([{"a": 1}]) == {"a": [1]}


if __name__ == "__main__":
    unittest.main(verbosity=2)
