# Derived anatomy validation

These reports contain computed errors, fitted factors, aggregate geometry
descriptors and provenance. Source coordinates, meshes, shape-model arrays and
archives are not included. Full alignment/residual evidence stays local because
it can reconstruct source coordinates when combined with a reference.

- `keast_landmark_validation.json`: CC BY 4.0. Derived from Keast, Bonacci and
  Fox, [Figshare version 4](https://doi.org/10.6084/m9.figshare.20454462.v4),
  including the independently acquired Nolte MRI cohort. The report contains
  attribution and protocol limits; it does not reproduce the source SSM study.
- `vsd_tibia_population_summary.json`: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  Derived from [BoneHub VSD](https://huggingface.co/datasets/BoneHub/vsd-lower-extremities-seg)
  and Fischer's source models. Its terms apply to this derivative report,
  separately from the Apache-2.0 implementation. Only aggregate descriptors and
  source hashes are retained; the cohort is not a general population prior.

The scripts reproduce these results from separately acquired local inputs.
Synthetic unit tests do not require those inputs. A fit with lower training
error is not necessarily a better prediction of withheld anatomy.
