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

scram b -j8
echo "SETUP_OK $BASE/$RELEASE"
