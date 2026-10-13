#!/bin/bash
# Create a CMSSW area and build ISpy in it. Idempotent: re-running only rebuilds.
# Runs inside the era's container (run_agent.py calls it via `cmssw-elN -- bash ...`).
#
#   bash setup_cmssw.sh <base dir> <CMSSW release> <SCRAM_ARCH> <ispy-analyzers branch> <ispy-services branch>

set -e
BASE=$1; RELEASE=$2; ARCH=$3; ISPY_BRANCH=$4; SERVICES_BRANCH=$5
if [ -z "$SERVICES_BRANCH" ]; then
  echo "usage: setup_cmssw.sh <base> <release> <arch> <ispy-branch> <services-branch>"; exit 1
fi

export SCRAM_ARCH=$ARCH
source /cvmfs/cms.cern.ch/cmsset_default.sh
mkdir -p "$BASE" && cd "$BASE"

if [ ! -d "$RELEASE" ]; then
  scram project CMSSW "$RELEASE"          # = cmsrel; the release itself comes from cvmfs
fi
cd "$RELEASE/src"
eval "$(scram runtime -sh)"             # = cmsenv

clone() {  # clone <repo> <branch> <dir>; refuse to silently use a different branch
  if [ ! -d "$3" ]; then
    git clone -b "$2" "https://github.com/cms-outreach/$1.git" "$3"
  fi
  local have
  have=$(git -C "$3" rev-parse --abbrev-ref HEAD)
  if [ "$have" != "$2" ]; then
    echo "[!] $3 is on branch '$have', expected '$2'. Remove $PWD/$3 to re-clone."; exit 1
  fi
  echo "$3: $have $(git -C "$3" log -1 --format='%h %ad' --date=short)"
}
clone ispy-analyzers "$ISPY_BRANCH" ISpy/Analyzers
clone ispy-services "$SERVICES_BRANCH" ISpy/Services

# ISpyMuon (cmssw10) fetches GEMGeometry with getData(), which throws NoProxyException when the
# job has no GEM geometry (e.g. 2016/2018 conditions). getHandle() returns an invalid handle
# instead; the module already handles a null gemGeometry_. Idempotent.
MUON=ISpy/Analyzers/src/ISpyMuon.cc
if grep -q 'gemGeometry_ *= *&eventSetup.getData(gemGeometryToken_);' "$MUON"; then
  sed -i 's|gemGeometry_ *= *&eventSetup.getData(gemGeometryToken_);|{ auto gemH = eventSetup.getHandle(gemGeometryToken_); gemGeometry_ = gemH.isValid() ? gemH.product() : nullptr; }|' "$MUON"
  echo "patched $MUON (optional GEM geometry)"
fi

scram b -j8
echo "SETUP_OK $BASE/$RELEASE"
