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


class FixtureTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + ord(character) % 29 for character in text]


@unittest.skipIf(torch is None, 'requires the pinned training packages for llm_data')
class LongDocumentPrefixTests(unittest.TestCase):
    """A minimum above the sequence length keeps each long document's first positions."""

    documents = [{'text': text, 'id': str(index)} for index, text in enumerate(
        ['abc', 'abcdefghijk', 'abcdefg', 'zyxwvutsrq', 'short', 'qrstuvwx', 'mnopqrstuvwxyz',
         'abcdefghij', 'tiny', 'klmnopqrstu', 'hgfedcbaxyz'])]

    def write(self, root, *, records, sequence_length):
        from llm_data import write_documents
        return write_documents(root, iter(self.documents), FixtureTokenizer(), records=records,
                               sequence_length=sequence_length, provenance={},
                               minimum_document_tokens=8)

    def rows(self, root, sequence_length):
        import numpy as np
        tokens = np.fromfile(Path(root) / 'tokens.bin', dtype='<i4').reshape(-1, sequence_length)
        lengths = np.fromfile(Path(root) / 'lengths.bin', dtype='<i4')
        return tokens, lengths

    def test_minimum_above_sequence_length_selects_long_documents_and_keeps_first_positions(self):
        tokenizer = FixtureTokenizer()
        long_documents = [d for d in self.documents if len(d['text']) >= 8]
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.write(Path(directory) / 'tokens', records=4, sequence_length=4)
            tokens, lengths = self.rows(Path(directory) / 'tokens', 4)
        self.assertEqual(manifest['minimum_document_tokens'], 8)
        self.assertEqual(lengths.tolist(), [4, 4, 4, 4])
        self.assertEqual(manifest['useful_tokens'], 4 * 3)
        for row, document in zip(tokens.tolist(), long_documents):
            self.assertEqual(row, tokenizer.encode(document['text'])[:4])

    def test_first_rows_do_not_depend_on_record_count(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write(Path(directory) / 'six', records=6, sequence_length=4)
            self.write(Path(directory) / 'three', records=3, sequence_length=4)
            six = (Path(directory) / 'six' / 'tokens.bin').read_bytes()
            three = (Path(directory) / 'three' / 'tokens.bin').read_bytes()
            six_lengths = (Path(directory) / 'six' / 'lengths.bin').read_bytes()
            three_lengths = (Path(directory) / 'three' / 'lengths.bin').read_bytes()
        self.assertEqual(six[:len(three)], three)
        self.assertEqual(six_lengths[:len(three_lengths)], three_lengths)

    def test_shorter_rows_are_the_first_positions_of_longer_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write(Path(directory) / 'long', records=6, sequence_length=8)
            self.write(Path(directory) / 'short', records=6, sequence_length=4)
            long_tokens, long_lengths = self.rows(Path(directory) / 'long', 8)
            short_tokens, short_lengths = self.rows(Path(directory) / 'short', 4)
        self.assertEqual(long_lengths.tolist(), [8] * 6)
        self.assertEqual(short_lengths.tolist(), [4] * 6)
        self.assertEqual(short_tokens.tolist(), long_tokens[:, :4].tolist())

    def test_negative_minimum_is_rejected(self):
        from llm_data import write_documents
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'nonnegative'):
            write_documents(directory, iter(self.documents), FixtureTokenizer(), records=1,
                            sequence_length=4, provenance={}, minimum_document_tokens=-1)


if __name__=='__main__': unittest.main()
