import io
import sys
import unittest

from adept.utils.requires_args import RequiresArgsMixin


class Stub(RequiresArgsMixin):
    args = {"k1": 0, "k2": False, "k3": 1.5, "k4": "hello"}


class TestRequiresArgs(unittest.TestCase):
    stub = Stub()

    def test_prompt_no_changes(self):
        sys.stdin = io.StringIO("\n")
        new_conf = self.stub.prompt()
        assert new_conf == self.stub.args

    def test_prompt_modify(self):
        sys.stdin = io.StringIO('{"k1": 5}')
        new_conf = self.stub.prompt()
        assert new_conf["k1"] == 5
        assert new_conf["k2"] == self.stub.args["k2"]
        assert new_conf["k3"] == self.stub.args["k3"]
        assert new_conf["k4"] == self.stub.args["k4"]


if __name__ == "__main__":
    unittest.main(verbosity=1)
