#!/bin/bash
# HTCondor job (CPU): pack npz into train/val/test (ml/pack.py).
# args: CODE_DIR PACK_OUT TARGET... [ml.pack options]
# Writes to the job's scratch disk, then copies each file to EOS with xrdcp.
set -euo pipefail
code=$1; out=$2; shift 2
view=${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_107_cuda/x86_64-el9-gcc11-opt}
[ -f "$view/setup.sh" ] || { echo "no LCG view $view (set lcg=... at submit)"; exit 1; }
set +u; source "$view/setup.sh"; set -u
echo "host $(hostname)  python $(python3 --version 2>&1)  numpy $(python3 -c 'import numpy; print(numpy.__version__)')"
cd "$code"
tmp=${_CONDOR_SCRATCH_DIR:-$(mktemp -d)}/pack
python3 -m ml.pack "$@" --out "$tmp"
cd "$tmp"
find . -type f | while read -r f; do
    xrdcp -f -p "$f" "root://eosuser.cern.ch/$out/${f#./}"
done
echo "copied to $out"
