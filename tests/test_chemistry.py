from lcms2smiles.chemistry import canonical_smiles, exact_mass, molecular_formula, neutral_mass_from_precursor


def test_chemistry_helpers():
    assert canonical_smiles("OCC") == "CCO"
    assert molecular_formula("CCO") == "C2H6O"
    assert 46.0 < exact_mass("CCO") < 47.0
    assert abs(neutral_mass_from_precursor(47.049141, "[M+H]+") - 46.041865) < 1e-3
