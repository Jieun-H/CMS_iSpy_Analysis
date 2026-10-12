#!/bin/bash
# Dump the event content of one input file and check that the collections the
# point cloud needs are there. Runs inside the era's container.
#
#   bash check_content.sh <CMSSW area> <file URL> <output txt>
#
# Exit codes: 0 ok, 2 missing collections, 3 looks like MiniAOD

set -e
AREA=$1; URL=$2; OUT=$3
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$AREA/src" && eval "$(scram runtime -sh)" && cd - >/dev/null

edmDumpEventContent "$URL" > "$OUT"
echo "event content -> $OUT"

if grep -qE '"(slimmedMuons|packedPFCandidates)"' "$OUT"; then
  echo "[!] slimmedMuons/packedPFCandidates found: this is MiniAOD, the AOD cfg would produce empty ig files."
  exit 3
fi

# label:instance pairs used by ispy_aod_cfg.py for the point cloud
REQUIRED="reducedEcalRecHitsEB: reducedEcalRecHitsEE: reducedHcalRecHits:hbhereco
trackExtrapolator: gedGsfElectrons: muons: offlinePrimaryVertices: inclusiveSecondaryVertices:"
MISSING=0
for r in $REQUIRED; do
  label=${r%%:*}; inst=${r#*:}
  if grep -qE "\"$label\" +\"$inst\"" "$OUT"; then
    printf "  [OK]      %s\n" "$r"
  else
    printf "  [MISSING] %s\n" "$r"; MISSING=1
  fi
done
[ $MISSING -eq 0 ] || exit 2
echo "CONTENT_OK"
