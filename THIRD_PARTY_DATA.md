# Third-party body-model and motion sources

The independently authored integration code and synthetic tests use the
repository's Apache-2.0 licence. No SMPL/SKEL model assets, upstream implementation
code, shape or pose parameters, marker-definition tables, source motion samples,
native models or coordinate exports are distributed by this integration. The
adapters operate on separately acquired user inputs and optional upstream readers.
Their availability does not grant rights to those inputs or to generated assets.

## Published validation results

`docs/source/_static/scaling/native_validation_metrics.json` contains scientific
validation outcomes: aggregate errors and counts, evaluation settings, selected
frame indices, source hashes and attribution. Its accompanying documentation
describes the same experiment. It contains no model parameters, vertex or joint
coordinates, individual fitted scale vectors, trajectories or motion exports.
The SKEL and BioAmass runs use the same CMU01 subject and source motion; they are
two representations, not independent subject cohorts.

These numerical research results do not redistribute or relicense the source
models or datasets. This notice provides no blanket permission to redistribute
derived model assets, geometry, markers or motions. Any such artifact needs its
own source-rights review. Local full fitting evidence can include alignments and
residuals that reconstruct source coordinates when combined with a reference;
keep that evidence with the licensed inputs unless sharing is permitted.

## SKEL and BioAmass

Source: [SKEL project and BioAmass description](https://skel.is.tue.mpg.de/).
The [SKEL licence](https://skel.is.tue.mpg.de/license.html) limits use to specified
noncommercial purposes and prohibits redistribution of the licensed data and
software. Its public code is subject to those terms too. BioAmass supplies BSM
fits to SMPL sequences from AMASS, so users must check the relevant SKEL,
AMASS and original motion-source terms together. No licence in this repository
overrides those requirements.

Required scientific credit for the local evaluation:

- Marilyn Keller, Keenon Werling, Soyong Shin, Scott Delp, Sergi Pujades,
  C. Karen Liu and Michael J. Black (2023), *From Skin to Skeleton: Towards
  Biomechanically Accurate 3D Digital Humans*, ACM Transactions on Graphics,
  [doi:10.1145/3618381](https://doi.org/10.1145/3618381).

## SMPL and AMASS

- Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll and Michael J.
  Black (2015), *SMPL: A Skinned Multi-Person Linear Model*, ACM Transactions on
  Graphics, [doi:10.1145/2816795.2818013](https://doi.org/10.1145/2816795.2818013).
  The [SMPL-Model licence](https://smpl.is.tue.mpg.de/modellicense.html) and
  [SMPL-Body licence](https://smpl.is.tue.mpg.de/bodylicense.html) cover different
  materials. The latter's CC BY 4.0 scope does not relicense the full model,
  shape tools, SKEL or AMASS. The integration supplies none of these assets.
- Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll and Michael J.
  Black (2019), *AMASS: Archive of Motion Capture as Surface Shapes*, ICCV,
  [project and citation](https://amass.is.tue.mpg.de/).
  The [AMASS licence](https://amass.is.tue.mpg.de/license.html) restricts use and
  redistribution independently of the original motion sources.

CMU source-motion credit: the original motion-capture data were obtained from
[mocap.cs.cmu.edu](http://mocap.cs.cmu.edu/), a database created with funding from
NSF EIA-0196217. Its [source policy](http://mocap.cs.cmu.edu/faqs.php) does not
override the separate licences for AMASS, BioAmass or SKEL representations.

## AddBiomechanics and Lencioni measured data

The [AddBiomechanics reviewed core 1.0](https://addbiomechanics.org/download_data.html)
is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The measured-marker experiment used the Lencioni/Tiziana source represented in
that release. All original `.b3d` data remain local; the public report retains
only computed validation outcomes, provenance and protocol settings.

Credit both the processing/dataset work and the original collection:

- Keenon Werling et al. (2023), *AddBiomechanics: Automating model scaling,
  inverse kinematics, and inverse dynamics from human motion data through
  sequential optimization*, PLOS ONE,
  [doi:10.1371/journal.pone.0295152](https://doi.org/10.1371/journal.pone.0295152).
- Keenon Werling et al., *AddBiomechanics Dataset: Capturing the Physics of
  Human Motion at Scale*, ECCV 2024 proceedings (published 2025),
  [doi:10.1007/978-3-031-73223-2_27](https://doi.org/10.1007/978-3-031-73223-2_27).
  See the release's [citation instructions](https://addbiomechanics.org/faq.html#citation).
- Tiziana Lencioni, Ilaria Carpinella, Marco Rabuffetti, Alberto Marzegan and
  Maurizio Ferrarin (2019), *Human kinematic, kinetic and EMG data during
  different walking and stair ascending and descending tasks*, Scientific Data
  6:309, [doi:10.1038/s41597-019-0323-z](https://doi.org/10.1038/s41597-019-0323-z).
  The publication's Data Records section specifies CC BY 4.0 for the original
  deidentified files.

## Runtime provenance and source notices

Callers provide source/version identity, units, coordinate transforms and
definition terms. These strings record provenance and do not certify access or
redistribution permission. Runtime `MarkerTrial` objects and saved fits can
contain user-provided coordinates, marker mappings and subject metadata; they
are local workflow artifacts, not automatically publishable outputs.

Sources are supplied without warranties under their respective licences. See
the SKEL and AMASS licence links above and the CC BY 4.0
[legal terms and disclaimer](https://creativecommons.org/licenses/by/4.0/legalcode.en).
The original authors and institutions do not endorse this integration or the
reported outcomes. Retain their applicable source notices and citations when
publishing research using their resources.
