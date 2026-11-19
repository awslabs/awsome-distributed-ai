"""Build a per-residue (character-level) protein tokenizer for OLMoE pretraining.

Vocabulary: 2 special tokens + 26 residue codes (20 canonical amino acids plus
X/B/Z/J/U/O extended codes found in UniProt/UniRef). Implemented as a BPE model
with an empty merge list, which degenerates to character-level tokenization —
this keeps the artifact loadable by `olmo.tokenizer.Tokenizer` (HF `tokenizers`
JSON format) without any code changes.

Usage:
    python scripts/build_protein_tokenizer.py
Writes olmo_data/tokenizers/protein_aa_v1.json and prints a round-trip check.
"""

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing

EOS = "<|endoftext|>"  # id 0, matches config eos_token_id
PAD = "<|padding|>"    # id 1, matches config pad_token_id

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
EXTENDED = "XBZJUO"  # unknown, Asp/Asn, Glu/Gln, Leu/Ile, selenocysteine, pyrrolysine

vocab = {EOS: 0, PAD: 1}
for residue in CANONICAL + EXTENDED:
    vocab[residue] = len(vocab)

# Empty merges -> pure character-level splitting. Unknown characters map to X
# rather than crashing, since stray codes do occur in real sequence data.
tokenizer = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="X"))
tokenizer.add_special_tokens([EOS, PAD])
tokenizer.post_processor = TemplateProcessing(
    single="$A " + EOS,
    special_tokens=[(EOS, vocab[EOS])],
)
# Fuse decoded tokens back into a contiguous sequence ("MKT", not "M K T")
tokenizer.decoder = decoders.Fuse()

if __name__ == "__main__":
    out = "olmo_data/tokenizers/protein_aa_v1.json"
    tokenizer.save(out)
    print(f"wrote {out} with vocab size {tokenizer.get_vocab_size()}")

    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    enc = tokenizer.encode(seq)
    decoded = tokenizer.decode(enc.ids, skip_special_tokens=True)
    print(f"ids:     {enc.ids}")
    print(f"tokens:  {enc.tokens}")
    assert decoded.replace(" ", "") == seq, f"round-trip failed: {decoded!r}"
    print("round-trip OK")
