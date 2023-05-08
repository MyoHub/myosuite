conda create --name test_myosuite python=3.7.1 -y
conda activate test_myosuite
pip install myosuite
python3 -c "import myosuite"
python3 myosuite/tests/test_myo.py
conda deactivate
conda remove --name test_myosuite --all -y
