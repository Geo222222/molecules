from lcms2smiles.tokenizer import SmilesTokenizer


def test_smiles_roundtrip():
    tok = SmilesTokenizer()
    s = "CC(=O)N[C@@H](CCl)C1=CC=CC=C1"
    assert tok.decode(tok.encode(s)) == s
