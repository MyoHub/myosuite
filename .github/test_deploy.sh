conda create --name test_myosuite python=3.10 -y
conda activate test_myosuite
pip install myosuite
python3 -c "import myosuite"
python3 -m myosuite.tests.test_myo
conda deactivate
conda remove --name test_myosuite --all -y
