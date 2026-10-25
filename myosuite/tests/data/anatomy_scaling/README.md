# Derived anatomy validation

These reports contain computed errors, fitted factors, aggregate geometry
descriptors and provenance. Source coordinates, meshes, shape-model arrays and
archives are not included. Full alignment/residual evidence stays local because
it can reconstruct source coordinates when combined with a reference.

- `keast_landmark_validation.json`: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Derived from Keast, Bonacci and
  Fox, [Figshare version 4](https://doi.org/10.6084/m9.figshare.20454462.v4),
  including the independently acquired Nolte MRI cohort. The report contains
  attribution and protocol limits; it does not reproduce the source SSM study.
  The corresponding `docs/source/_static/anatomy_scaling/keast_validation.png`
  is a new plot of these computed errors, also under CC BY 4.0.
- `vsd_tibia_population_summary.json`: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  Derived from [BoneHub VSD](https://huggingface.co/datasets/BoneHub/vsd-lower-extremities-seg)
  and Fischer's source models. Its terms apply to this derivative report,
  separately from the Apache-2.0 implementation. Only aggregate descriptors and
  source hashes are retained; the cohort is not a general population prior.

These identified reports and plot are exceptions to the implementation's
Apache-2.0 licence. Retain attribution, licence links and the description of
changes when sharing them. Sources and reports are supplied without warranties;
see the [CC BY legal terms](https://creativecommons.org/licenses/by/4.0/legalcode.en)
and [CC BY-NC-SA legal terms](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en).
Complete source citations, provenance and change descriptions appear in
[`THIRD_PARTY_DATA.md`](../../../../THIRD_PARTY_DATA.md). No endorsement is implied.

The scripts reproduce these results from separately acquired local inputs.
Synthetic unit tests do not require those inputs. A fit with lower training
error is not necessarily a better prediction of withheld anatomy.
