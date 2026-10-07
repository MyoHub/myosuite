# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Central citation metadata for optional integrations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegrationCitation:
    """Citation metadata exposed for docs and package-level discovery."""

    key: str
    name: str
    project_url: str
    bibtex: str
    arxiv_url: str | None = None


MUSCLEMIMIC_CITATION = IntegrationCitation(
    key="musclemimic",
    name="MuscleMimic",
    project_url="https://github.com/amathislab/musclemimic",
    arxiv_url="https://arxiv.org/abs/2603.25544",
    bibtex="""@article{Li2026MuscleMimic,
  title={Towards Embodied AI with MuscleMimic: Unlocking full-body musculoskeletal motor learning at scale},
  author={Li, Chengkun and Wang, Cheryl and Ziliotto, Bianca and Simos, Merkourios and Kovecses, Jozsef and Durandau, Guillaume and Mathis, Alexander},
  journal={arXiv preprint arXiv:2603.25544},
  year={2026}
}""",
)

AMASS_CITATION = IntegrationCitation(
    key="amass",
    name="AMASS",
    project_url="https://amass.is.tue.mpg.de",
    arxiv_url="https://arxiv.org/abs/1904.03278",
    bibtex="""@inproceedings{Mahmood2019AMASS,
  title={{AMASS}: Archive of Motion Capture as Surface Shapes},
  author={Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F. and Pons-Moll, Gerard and Black, Michael J.},
  booktitle={International Conference on Computer Vision (ICCV)},
  pages={5442--5451},
  year={2019}
}""",
)

KIT_MOTION_DATABASE_CITATION = IntegrationCitation(
    key="kit",
    name="KIT whole-body human motion database",
    project_url="https://motion-database.humanoids.kit.edu",
    bibtex="""@inproceedings{Mandery2015KIT,
  title={The {KIT} Whole-Body Human Motion Database},
  author={Mandery, Christian and Terlemez, {\\"O}mer and Do, Martin and Vahrenkamp, Nikolaus and Asfour, Tamim},
  booktitle={International Conference on Advanced Robotics (ICAR)},
  pages={329--336},
  year={2015}
}

@article{Mandery2016KIT,
  title={Unifying Representations and Large-Scale Whole-Body Motion Databases for Studying Human Motion},
  author={Mandery, Christian and Terlemez, {\\"O}mer and Do, Martin and Vahrenkamp, Nikolaus and Asfour, Tamim},
  journal={IEEE Transactions on Robotics},
  volume={32},
  number={4},
  pages={796--809},
  year={2016},
  doi={10.1109/TRO.2016.2572685}
}""",
)

GMR_CITATION = IntegrationCitation(
    key="gmr",
    name="General Motion Retargeting (GMR)",
    project_url="https://github.com/YanjieZe/GMR",
    arxiv_url="https://arxiv.org/abs/2510.02252",
    bibtex="""@article{Araujo2025GMR,
  title={Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking},
  author={Ara{\\'u}jo, Jo{\\~a}o Pedro and Ze, Yanjie and Xu, Pei and Wu, Jiajun and Liu, C. Karen},
  journal={arXiv preprint arXiv:2510.02252},
  year={2025}
}""",
)

KINESIS_CITATION = IntegrationCitation(
    key="kinesis",
    name="KINESIS",
    project_url="https://github.com/amathislab/Kinesis",
    arxiv_url="https://arxiv.org/abs/2503.14637",
    bibtex="""@article{Simos2025KINESIS,
  title={{KINESIS}: Motion Imitation for Human Musculoskeletal Locomotion},
  author={Simos, Merkourios and Chiappa, Alberto Silvio and Mathis, Alexander},
  journal={arXiv preprint arXiv:2503.14637},
  year={2025}
}""",
)

INTEGRATION_CITATIONS = (
    MUSCLEMIMIC_CITATION,
    AMASS_CITATION,
    KIT_MOTION_DATABASE_CITATION,
    GMR_CITATION,
    KINESIS_CITATION,
)
INTEGRATION_CITATIONS_BY_KEY = {
    citation.key: citation for citation in INTEGRATION_CITATIONS
}


def get_integration_citation(key: str) -> IntegrationCitation:
    """Return citation metadata for an integration key."""
    try:
        return INTEGRATION_CITATIONS_BY_KEY[key]
    except KeyError as exc:
        known = ", ".join(sorted(INTEGRATION_CITATIONS_BY_KEY))
        raise KeyError(f"Unknown integration citation {key!r}. Known: {known}") from exc


__all__ = [
    "AMASS_CITATION",
    "GMR_CITATION",
    "INTEGRATION_CITATIONS",
    "INTEGRATION_CITATIONS_BY_KEY",
    "IntegrationCitation",
    "KINESIS_CITATION",
    "KIT_MOTION_DATABASE_CITATION",
    "MUSCLEMIMIC_CITATION",
    "get_integration_citation",
]
