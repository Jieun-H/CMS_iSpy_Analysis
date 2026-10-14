#!/bin/bash
# HTCondor job (GPU): train ParticleNet (ml/train.py).
# args: CODE_DIR PACK RUN_OUT [ml.train options]
# The pack is copied from EOS to the job's scratch disk (fast reads every epoch); the run
# folder (checkpoints, metrics) is written straight to EOS so a killed job keeps last.pt.
set -euo pipefail
code=$1; pack=$2; out=$3; shift 3
view=${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_107_cuda/x86_64-el9-gcc11-opt}
[ -f "$view/setup.sh" ] || { echo "no LCG view $view (set lcg=... at submit)"; exit 1; }
set +u; source "$view/setup.sh"; set -u
echo "host $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
local_pack=${_CONDOR_SCRATCH_DIR:-$(mktemp -d)}/pack
trap 'rm -rf "$local_pack"' EXIT            # never sent back with the job output
mkdir -p "$local_pack"
cp -r "$pack"/. "$local_pack"/
echo "pack copied: $(du -sh "$local_pack" | cut -f1)"
mkdir -p "$out"
cd "$code"
resume=""
[ -f "$out/last.pt" ] && resume="--resume" && echo "found $out/last.pt, resuming"
python3 -m ml.train --pack "$local_pack" --pack-origin "$pack" --out "$out" $resume "$@"
