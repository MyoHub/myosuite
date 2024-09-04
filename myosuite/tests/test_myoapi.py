
from pathlib import Path
from unittest.mock import patch

from myosuite_init import fetch_simhive

my_file = Path("/path/to/file")

@patch('builtins.input', side_effect=['no'])
def test_no_myoapi(mock_input):
    print("mock_input", mock_input.side_effect)
    fetch_simhive()
    assert not Path("myosuite/simhive/myo_model/myoskeleton/myoskeleton.xml").exists()

@patch('builtins.input', side_effect=['yes'])
def test_yes_myoapi(mock_input):
    print("mock_input", mock_input.side_effect)
    fetch_simhive()
    assert Path("myosuite/simhive/myo_model/myoskeleton/myoskeleton.xml").exists()

test_no_myoapi()
test_yes_myoapi()



