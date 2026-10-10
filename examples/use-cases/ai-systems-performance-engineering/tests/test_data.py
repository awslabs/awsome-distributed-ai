"""Both layouts must feed identical tokens, including the final partial shard."""
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Run inside the pinned training image for dataset checks')
class LayoutTests(unittest.TestCase):
    def test_layout_equivalence(self):
        from data import prepare, Tokens
        with tempfile.TemporaryDirectory() as directory:
            prepare(directory,records=7,sequence=4,per_shard=3,vocab=32,metadata_checks=2)
            small=Tokens(directory,'small'); packed=Tokens(directory,'shards')
            self.assertEqual(len(small),7)
            for index in range(len(small)):
                self.assertTrue(torch.equal(small[index],packed[index]))


if __name__=='__main__': unittest.main()
