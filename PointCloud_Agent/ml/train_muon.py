"""
Train the per-track muon label (tracker muon match) with ParticleNet: ml/train.py with
--particle muon. All other options are those of ml/train.py (python3 -m ml.train_muon -h).

    python3 -m ml.train_muon --pack /eos/user/j/jieun/ml/packed/zmm_ul16_v1 --out RUN_DIR
    python3 -m ml.train_muon --pack PACK --out RUN --hit-keys ecal_eb ecal_ee
    python3 -m ml.train_muon --pack PACK --out RUN --task event        # has a global muon
    # condor: ml/condor/train_muon.sub
"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import train  # noqa: E402

PARTICLE = "muon"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a == "--particle" or a.startswith("--particle=") for a in argv):
        raise SystemExit(f"{os.path.basename(__file__)} always trains --particle {PARTICLE}")
    return train.main(["--particle", PARTICLE] + argv)


if __name__ == "__main__":
    sys.exit(main())
