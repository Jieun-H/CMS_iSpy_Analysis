"""
Era table: which CMSSW release, container, ISpy branch and GlobalTag to use for a dataset.

The CMSSW release that produced a dataset is the source of truth (DAS `release dataset=...`).
iSpy should run in a release as close as possible to that one
(https://github.com/cms-outreach/ispy-analyzers/blob/cmssw12/README.md, "Note").
"""

import re
from typing import List, Optional


class Era(object):
    """Plain class (no dataclasses): must run on python3.6 inside CMSSW_10_6 / el7."""

    def __init__(self, key, description, release_prefixes, cmssw, scram_arch, container,
                 ispy_branch, services_branch, data_globaltag, validated, notes=""):
        self.key = key
        self.description = description
        self.release_prefixes = release_prefixes  # DAS releases handled, e.g. ("CMSSW_10_6_",)
        self.cmssw = cmssw                        # release to cmsrel
        self.scram_arch = scram_arch
        self.container = container                # cmssw-elN wrapper on lxplus9, None = el9 host
        self.ispy_branch = ispy_branch            # cms-outreach/ispy-analyzers
        self.services_branch = services_branch    # cms-outreach/ispy-services
        self.data_globaltag = data_globaltag      # default GlobalTag for collision data
        self.validated = validated                # False = never run end-to-end
        self.notes = notes

    def matches(self, release):
        return any(release.startswith(p) for p in self.release_prefixes)


ERAS: List[Era] = [
    Era(key="run2_ul",
        description="Run2 Ultra-Legacy (106X) AOD/AODSIM",
        release_prefixes=("CMSSW_10_6_",),
        cmssw="CMSSW_10_6_30", scram_arch="slc7_amd64_gcc700", container="cmssw-el7",
        ispy_branch="cmssw10", services_branch="master",
        data_globaltag="auto:run2_data", validated=False,
        notes="Uses the ISpy cmssw10 branch, which has no CI for 10_6 (unvalidated)."),
    Era(key="run3_120x",
        description="Run3 Summer21 MC (120X) AODSIM",
        release_prefixes=("CMSSW_12_0_", "CMSSW_12_1_"),
        cmssw="CMSSW_12_0_4", scram_arch="slc7_amd64_gcc900", container="cmssw-el7",
        ispy_branch="cmssw12", services_branch="master",
        data_globaltag="", validated=False,
        notes="cmssw12 branch CI builds CMSSW_12_0_0 / slc7_amd64_gcc900."),
    Era(key="run3_13x",
        description="Run3 2022-2023 (12_4X-13_X) AOD/AODSIM",
        release_prefixes=("CMSSW_12_4_", "CMSSW_12_6_", "CMSSW_13_"),
        cmssw="CMSSW_13_0_17", scram_arch="el8_amd64_gcc11", container="cmssw-el8",
        ispy_branch="cmssw13", services_branch="master",
        data_globaltag="", validated=False,
        notes="Experimental: no CI for cmssw13. Pass --globaltag for collision data."),
    Era(key="run3_14x",
        description="Run3 2024+ (14_X) AOD/AODSIM",
        release_prefixes=("CMSSW_14_",),
        cmssw="CMSSW_14_0_15", scram_arch="el9_amd64_gcc12", container=None,
        ispy_branch="cmssw14", services_branch="master",
        data_globaltag="", validated=False,
        notes="Experimental: no CI for cmssw14. Global muon ptMin default 2 GeV."),
]

# Processing-string prefix in the dataset name -> release prefix, used when DAS gives no release.
_NAME_TO_RELEASE = [
    (re.compile(r"-106X_"), "CMSSW_10_6_"),
    (re.compile(r"-12[01]X_"), "CMSSW_12_0_"),
    (re.compile(r"-12[4-6]X_"), "CMSSW_12_4_"),
    (re.compile(r"-13\dX_"), "CMSSW_13_0_"),
    (re.compile(r"-14\dX_"), "CMSSW_14_0_"),
]


def parse_dataset(dataset: str) -> dict:
    """/Primary/Processed/TIER -> parts; raises ValueError for anything else."""
    parts = dataset.strip().split("/")
    if len(parts) != 4 or parts[0] != "" or not all(parts[1:]):
        raise ValueError(f"Not a DAS dataset name (/Primary/Processed/TIER): {dataset}")
    primary, processed, tier = parts[1:]
    return {"primary": primary, "processed": processed, "tier": tier,
            "is_mc": tier.endswith("SIM")}


def mc_globaltag(dataset: str) -> Optional[str]:
    """
    MC GlobalTag embedded in the processed dataset name:
      RunIISummer20UL16RECO-106X_mcRun2_asymptotic_v13-v1 -> 106X_mcRun2_asymptotic_v13
      Run3Summer21DRPremix-120X_mcRun3_2021_realistic_v6-v2 -> 120X_mcRun3_2021_realistic_v6
    """
    processed = parse_dataset(dataset)["processed"]
    m = re.match(r"^[^-]+-(?:pilot_)?(\d+X_.+?)-v\d+$", processed)
    if not m:
        return None
    return re.sub(r"_ext\d+$", "", m.group(1))     # ..._L1v1_ext1 is a dataset extension, not part of the GT


def release_from_name(dataset: str) -> Optional[str]:
    for pattern, prefix in _NAME_TO_RELEASE:
        if pattern.search(dataset):
            return prefix
    return None


def era_for_release(release: str) -> Era:
    for era in ERAS:
        if era.matches(release):
            return era
    known = ", ".join(p for e in ERAS for p in e.release_prefixes)
    raise ValueError(f"No era configured for {release} (known: {known}). "
                     f"Run1 (CMSSW_5_3_X) uses a different cfg — see ig/Generate_ig/Run1_AOD.")


def default_name(dataset: str) -> str:
    """Output name: Primary + full processed name (keeps era/processing version/-vN apart)."""
    p = parse_dataset(dataset)
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{p['primary']}__{p['processed']}")


# SCRAM_ARCH prefix -> OS major version it needs (cmssw-elN on lxplus9)
_ARCH_OS = {"slc6": 6, "slc7": 7, "cc8": 8, "el8": 8, "el9": 9, "el10": 10}


def arch_os(scram_arch: str) -> Optional[int]:
    return _ARCH_OS.get(scram_arch.split("_")[0])
