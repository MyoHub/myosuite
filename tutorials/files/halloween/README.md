# Halloween teaching data

Small pretrained data files for the explicit Python/Colab lesson. No executable Python is embedded in either artifact.

- `motion_decoder.npz`: command/phase RBF coefficients, frame counts, and the settled initial qpos/qvel/activation state. SHA-256 `b3f9a27ed5a9b3dd64f0efb8671eb5954fe1c0ce1ed681e3e5ba339d2468b78c`.
- `selector.joblib`: scikit-learn ExtraTreesRegressor, trained to score idle/left/right/squat from projectile intercept features and command one-hot values. SHA-256 `c748be2fa397cd8aefb70bf7fa6eda20eb5e5ecefb49963245531cb41097db8e`. Load only this trusted, verified file; joblib uses pickle.

These preserve the existing MyoHalloween `reactive_selector_v2` artifacts. TERRA neural-network weights are not vendored: the tutorial downloads the pinned public TERRA-4B checkpoint through MyoSuite. The tutorial uses the registered environment and a separate controller; the library owns physics and learned reference decoding. Keeping these teaching files under tutorials excludes them from the pip package.

## Follow the lesson

[Notebook](../../4.5_Halloween_Avoidance.ipynb) · [Explicit Python](../../4.5_Halloween_Avoidance.py) · [Local requirements](requirements-student.txt).

The lesson shows environment loading, seeded reset, controller loading, muscle-action stepping, rendering and quantitative checks. Task construction and contacts live in the environment; the controller reuses TERRA. No private repository access is required. For local use, download the Python file and requirements into one directory, create a Python 3.12 virtual environment, install the requirements and run the file. CPU with EGL is the verified rendering path; Colab is recommended on macOS/Windows. See [executed checks](verification.json).
