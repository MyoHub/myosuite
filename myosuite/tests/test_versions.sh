pip uninstall -y gym
pip uninstall -y gymnasium

# echo "=================== Testing gym==0.13 ==================="
pip install gym==0.13
python myosuite/tests/test_myo.py
pip uninstall -y gym

echo "=================== Testing gym==0.26.2 ==================="
pip install gym==0.26.2
python myosuite/tests/test_myo.py
pip uninstall -y gym

echo "=================== Testing gymnasium ==================="
pip install gymnasium
python myosuite/tests/test_myo.py
python myosuite/tests/test_sb.py
pip uninstall -y gymnasium
