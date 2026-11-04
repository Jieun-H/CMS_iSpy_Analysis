"""
Train the per-track electron label (GSF electron match) with ParticleNet: ml/train.py with
--particle electron. All other options are those of ml/train.py (python3 -m ml.train_electron -h).

The pack needs the electron arrays (packed with the current ml/pack.py, e.g. DYJetsToLL).
The label cuts --e-angle / --e-d0 are provisional: choose them with ml/electron_match.py
first. AOD keeps ECAL hits only around reconstructed electrons / photons, so the ECAL
inputs partly tell the model where reco found an electron; compare with --hit-keys (none).

    python3 -m ml.train_electron --pack /eos/user/j/jieun/ml/packed/dyll_ul16_v1 --out RUN_DIR
    python3 -m ml.train_electron --pack PACK --out RUN --e-angle 0.005 --e-d0 0.002
    python3 -m ml.train_electron --pack PACK --out RUN --hit-keys      # no calorimeter
    python3 -m ml.train_electron --pack PACK --out RUN --task event    # has a GSF electron
    # condor: ml/condor/train_electron.sub
"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import train  # noqa: E402

PARTICLE = "electron"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a == "--particle" or a.startswith("--particle=") for a in argv):
        raise SystemExit(f"{os.path.basename(__file__)} always trains --particle {PARTICLE}")
    return train.main(["--particle", PARTICLE] + argv)


if __name__ == "__main__":
    sys.exit(main())
