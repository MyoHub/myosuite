from pathlib import Path
from unittest.mock import patch

from myosuite_init import clean_simhive, fetch_simhive

TARGET_XML = (
    Path(__file__).resolve().parents[1]
    / "simhive"
    / "myo_model"
    / "myoskeleton"
    / "myoskeleton.xml"
)


@patch("builtins.input", side_effect=["no"])
def test_no_myoapi(mock_input):
    print("mock_input", mock_input.side_effect)
    clean_simhive()

    fetch_simhive()
    assert not TARGET_XML.exists()


@patch("builtins.input", side_effect=["yes"])
def test_yes_myoapi(mock_input):
    print("mock_input", mock_input.side_effect)
    clean_simhive()
    fetch_simhive()
    assert TARGET_XML.exists()


test_no_myoapi()
test_yes_myoapi()
