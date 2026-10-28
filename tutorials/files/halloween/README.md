# Halloween teaching data

Small pretrained data files for the explicit Python/Colab lesson. No executable Python is embedded in either artifact.

- `motion_decoder.npz`: command/phase RBF coefficients, frame counts, and the settled initial qpos/qvel/activation state. SHA-256 `b3f9a27ed5a9b3dd64f0efb8671eb5954fe1c0ce1ed681e3e5ba339d2468b78c`.
- `selector.joblib`: scikit-learn ExtraTreesRegressor, trained to score idle/left/right/squat from projectile intercept features and command one-hot values. SHA-256 `c748be2fa397cd8aefb70bf7fa6eda20eb5e5ecefb49963245531cb41097db8e`. Load only this trusted, verified file; joblib uses pickle.

These preserve the existing MyoHalloween `reactive_selector_v2` artifacts. TERRA neural-network weights are not vendored: the tutorial downloads the pinned public TERRA-4B checkpoint through MyoSuite. The tutorial displays inference, physics and rendering code explicitly. Keeping these teaching files under tutorials excludes them from the pip package.
