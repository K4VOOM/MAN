import math
from data_loader import load_parameters

def test_s0_phase1_values():
    params = load_parameters()
    assert params['S0_phase1'] == (16000, 17000, 17000), "S0_phase1 values are incorrect"

def test_input_values():
    params = load_parameters('input')
    assert 'val_Lr_killed' not in params, "val_Lr_killed should not be present in input parameters"

def test_validation_values():
    params = load_parameters('validation')
    assert 'val_Lr_killed' in params, "val_Lr_killed should be present in validation parameters"
    assert 'S0_phase1' not in params, "S0_phase1 should not be present in validation parameters"

def test_nan_values():
    params = load_parameters()['A_R']
    assert math.isnan(params[1]), "A_R likely should be NaN"