# Third-party anatomy sources and validation assets

The independently authored Python implementation and synthetic tests use the
repository's Apache-2.0 licence. That licence does **not** relicense the separately
identified validation assets below. No source meshes, landmark coordinates,
medical images, statistical-shape arrays or source archives are distributed.
Source URLs, file hashes and protocol labels identify separately acquired inputs;
they do not grant access or reuse rights to those inputs.

## Distributed validation assets

| Repository artifact | Licence applying to that artifact | Changes from its sources |
| --- | --- | --- |
| `myosuite/tests/data/anatomy_scaling/keast_landmark_validation.json` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Computed per-case scale factors and scalar fitting/withheld-landmark errors, paired summaries, protocol and input hashes. No source coordinates or model arrays. |
| `docs/source/_static/anatomy_scaling/keast_validation.png` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Plot of the computed error report above; no source anatomy rendering. |
| `myosuite/tests/data/anatomy_scaling/vsd_tibia_population_summary.json` | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | Aggregate geometric descriptors and quality coverage computed from 20 right-tibia meshes, plus input hashes. No per-case geometry measurements or source meshes. |
| `myosuite/tests/data/denver_geometry/validation.json` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Aggregate checks comparing closed-mesh volumes with independently published volume tables, plus quality counts and input hashes. No source tables, meshes or per-geometry volume rows. |

The VSD report retains the source's noncommercial and share-alike terms. Those
terms apply to that identified report and adaptations of it, independently of
the Apache-2.0 code. Calling something a derivative does not remove its source
licence obligations. Retain the relevant attribution, licence link and change
description when sharing any of these validation assets.

The sources and reports are provided without warranties; see the CC BY 4.0
[legal terms and disclaimer](https://creativecommons.org/licenses/by/4.0/legalcode.en)
and the CC BY-NC-SA 4.0
[legal terms and disclaimer](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en).
No source author or institution endorses this integration or its computed results.

## Keast tibia-fibula and Nolte MRI sources

- Meghan Keast, Jason Bonacci and Aaron Fox (2023), *Geometric variation of the
  human tibia-fibula: a public dataset of tibia-fibula surface meshes and
  statistical shape model*, PeerJ 11:e14708,
  [doi:10.7717/peerj.14708](https://doi.org/10.7717/peerj.14708).
- Keast, Bonacci and Fox, *Statistical Shape Model of the Tibia*,
  [Figshare version 4](https://doi.org/10.6084/m9.figshare.20454462.v4),
  [versioned licence metadata](https://api.figshare.com/v2/articles/20454462/versions/4),
  CC BY 4.0.
- Chui Kit Tsang, Kai Yu Zhang, Daniel Nolte, Angela E. Kedgley and Anthony M. J.
  Bull, original MRI-derived lower-limb geometry data,
  [doi:10.5281/zenodo.167808](https://doi.org/10.5281/zenodo.167808), CC BY 4.0.
  Credit also the archive's requested Nolte et al. (2016) publication,
  [doi:10.1016/j.jbiomech.2016.09.005](https://doi.org/10.1016/j.jbiomech.2016.09.005).

The validation computes its own six longitudinal border fractions and is not an
exact reproduction of the source reconstruction experiment. The source MATLAB
implementation and its bundled dependencies are not included. Original CT
medical images require separate access from their provider and are not inputs
bundled by this integration.

## BoneHub Visible Human and VSD sources

- Alavi and Asseln (2026), *BoneHub Visible Human 3D models*,
  [doi:10.57967/hf/10464](https://doi.org/10.57967/hf/10464),
  [dataset and licence notice](https://huggingface.co/datasets/BoneHub/visible-human-3d-models),
  CC BY 4.0. Credit also Andreassen et al. (2023), cited below.
- Alavi and Asseln (2026), *BoneHub VSD lower-extremity segmentations*,
  [doi:10.57967/hf/10471](https://doi.org/10.57967/hf/10471),
  [versioned dataset and licence notice](https://huggingface.co/datasets/BoneHub/vsd-lower-extremities-seg/blob/85214b5a182b980a7869b57efba89a59a2445875/README.md),
  CC BY-NC-SA 4.0. The dataset card's DOI heading is used; its BibTeX DOI has a
  different suffix. Credit also Fischer (2023),
  [doi:10.1038/s41597-023-02669-z](https://doi.org/10.1038/s41597-023-02669-z).
  The [original VSD v3 source notice](https://github.com/MCM-Fischer/VSDFullBodyBoneModels/blob/v3.0/README.md)
  independently identifies CC BY-NC-SA 4.0 for data and EUPL-1.2 for its source
  code. No upstream MATLAB or Python source code is included here.

Only the VSD aggregate report identified above is distributed. The selected
cadavers do not establish a representative population distribution; the report
is not an automatically applied subject prior. Downloaded assets stay in the
user's separately chosen local cache under their applicable source terms.

## University of Denver Visible Human tissue sources

The source archive READMEs specify CC BY 4.0 and request both publications:

- Andreassen et al. (2023),
  [Scientific Data, doi:10.1038/s41597-022-01905-2](https://doi.org/10.1038/s41597-022-01905-2).
- Andreassen et al., the accompanying overclosure study,
  [doi:10.48550/arXiv.2209.06948](https://doi.org/10.48550/arXiv.2209.06948).
- Dataset: University of Denver,
  [Visible Human data, doi:10.56902/COB.vh.2022.0](https://doi.org/10.56902/COB.vh.2022.0),
  [source and licence notice](https://digitalcommons.du.edu/visiblehuman/).
  Subject records: [female, doi:10.56902/COB.vh.2022.1](https://digitalcommons.du.edu/visiblehuman/1/)
  and [male, doi:10.56902/COB.vh.2022.2](https://digitalcommons.du.edu/visiblehuman/2/).

These are the same two donors represented by BoneHub, not two additional
population observations. The distributed report computes new consistency and
quality summaries; it does not redistribute source geometry or volume tables.
Retain the source records' Liability Agreement notice: the source is supplied
as is, without guarantees or warranties or assumed provider liability; consult
the linked source records for the complete notice.

## Local outputs

The adapters require explicit source identity, units and frame definitions.
User-supplied data, marker definitions and exported motions retain their own
terms. Full fitting evidence can contain alignments and point residuals that,
together with a reference, reconstruct observations; keep that evidence local
unless the corresponding source rights permit sharing it. Software access does
not establish permission to redistribute its inputs or outputs.
