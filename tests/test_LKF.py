import unittest
from deepmd_pt.utils.env import JIT
from deepmd_pt.entrypoints.main import main


class TestLKF(unittest.TestCase):
    def test_lkf(self):
        if JIT == True:
            return
        main(["train", "tests/water/lkf.json"])


if __name__ == '__main__':
    unittest.main()
