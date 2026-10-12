#!/bin/bash
# Run the ISpy cfg in a CMSSW area. Runs inside the era's container.
#
#   bash run_cmsrun.sh <CMSSW area> <cfg> <cmsRun arguments...>

set -e
AREA=$1; CFG=$2; shift 2
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$AREA/src" && eval "$(scram runtime -sh)" && cd - >/dev/null
cmsRun "$CFG" "$@"
