# -*- coding: utf-8 -*-
"""
AOD/AODSIM -> iSpy ig, for the Run2 UL and Run3 ISpy branches (cmssw10, cmssw12, cmssw13, cmssw14).

Written for both python2.7 (CMSSW_10_6 reads cfgs with python2) and python3: no f-strings.

Example:
  cmsRun ispy_aod_cfg.py inputFiles=/store/mc/.../file.root \
      globalTag=120X_mcRun3_2021_realistic_v6 \
      outputDir=/tmp/pca/NAME outputName=NAME maxEvents=100 eventsPerIg=10

inputFiles can be repeated, or given as a list file with inputFiles_load=files.txt.
"""

import os
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')   # provides inputFiles, maxEvents
options.register('globalTag', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "GlobalTag (MC: taken from the dataset name; data: auto:run2_data, ...)")
options.register('outputDir', os.getcwd(), VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Directory for the ig files (absolute path)")
options.register('outputName', 'ispy', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "ig file prefix -> <outputName>_0.ig, _1.ig, ...")
options.register('eventsPerIg', 10, VarParsing.multiplicity.singleton, VarParsing.varType.int,
                 "Events per ig file")
options.register('xrootdPrefix', 'root://cms-xrd-global.cern.ch/', VarParsing.multiplicity.singleton,
                 VarParsing.varType.string, "Redirector prepended to /store/... LFNs")
# ISpy defaults (match what the iSpy display shows). Set to 0 to keep low-pT objects.
options.register('trackPtMin', 1.0, VarParsing.multiplicity.singleton, VarParsing.varType.float,
                 "Track / GSF electron / tracker muon pT cut (GeV)")
options.register('muonPtMin', 2.0, VarParsing.multiplicity.singleton, VarParsing.varType.float,
                 "Global muon pT cut (GeV); only branches that support it (cmssw10, cmssw14+)")
options.maxEvents = 100
options.parseArguments()

if not options.globalTag:
    raise RuntimeError("globalTag=... is required")


def to_url(f):
    """Prefix /store/... LFNs with the xrootd redirector; leave root:// and file: untouched."""
    f = f.strip()
    if f.startswith('/store/'):
        return options.xrootdPrefix + '/' + f
    return f


output_dir = os.path.abspath(options.outputDir)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

process = cms.Process("ISPY")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

process.source = cms.Source(
    'PoolSource',
    fileNames=cms.untracked.vstring([to_url(f) for f in options.inputFiles if f.strip()])
)
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.maxEvents))

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 10

# Do not use outputFilePath: ISpyService also prepends it to the entry names inside the
# zip ("/out/dir/Events/Run_X/Event_Y" instead of "Events/Run_X/Event_Y").
# An absolute outputFileName gives the right file location and clean entry names.
process.add_(
    cms.Service("ISpyService",
        outputFileName   = cms.untracked.string(os.path.join(output_dir, options.outputName + '.ig')),
        outputESFileName = cms.untracked.string(os.path.join(output_dir, options.outputName + '_ES.ig')),
        outputIg         = cms.untracked.bool(True),
        outputMaxEvents  = cms.untracked.int32(options.eventsPerIg),
    )
)

process.options = cms.untracked.PSet(
    SkipEvent=cms.untracked.vstring('ProductNotFound')
)

# ── Analyzers ──────────────────────────────────────────────────────────────

ANALYZERS = [
    'ISpyEvent',
    # muon system (AOD only)
    'ISpyCSCRecHit2D', 'ISpyCSCSegment', 'ISpyDTRecHit', 'ISpyDTRecSegment4D', 'ISpyRPCRecHit',
    # calorimeters
    'ISpyEBRecHit', 'ISpyEERecHit', 'ISpyESRecHit',
    'ISpyHBRecHit', 'ISpyHERecHit', 'ISpyHFRecHit', 'ISpyHORecHit',
    # physics objects
    'ISpyMuon', 'ISpyPhoton', 'ISpyPFJet', 'ISpyPFMET',
    # Tracks_V3, GsfElectrons_V2, TrackerMuons_V2 + Extras_V1
    'ISpyTrackExtrapolation',
    # PrimaryVertices_V1, SecondaryVertices_V1
    'ISpyVertex',
]
for name in ANALYZERS:
    process.load('ISpy.Analyzers.%s_cfi' % name)

# ── InputTags (Run2 UL / Run3 AOD) ─────────────────────────────────────────

process.ISpyCSCRecHit2D.iSpyCSCRecHit2DTag      = cms.InputTag("csc2DRecHits")
process.ISpyCSCSegment.iSpyCSCSegmentTag         = cms.InputTag("cscSegments")
process.ISpyDTRecHit.iSpyDTRecHitTag            = cms.InputTag("dt1DRecHits")
process.ISpyDTRecSegment4D.iSpyDTRecSegment4DTag = cms.InputTag("dt4DSegments")
process.ISpyRPCRecHit.iSpyRPCRecHitTag          = cms.InputTag("rpcRecHits")

process.ISpyEBRecHit.iSpyEBRecHitTag = cms.InputTag("reducedEcalRecHitsEB")
process.ISpyEERecHit.iSpyEERecHitTag = cms.InputTag("reducedEcalRecHitsEE")
process.ISpyESRecHit.iSpyESRecHitTag = cms.InputTag("reducedEcalRecHitsES")

process.ISpyHBRecHit.iSpyHBRecHitTag = cms.InputTag("reducedHcalRecHits:hbhereco")
process.ISpyHERecHit.iSpyHERecHitTag = cms.InputTag("reducedHcalRecHits:hbhereco")
process.ISpyHFRecHit.iSpyHFRecHitTag = cms.InputTag("reducedHcalRecHits:hfreco")
process.ISpyHORecHit.iSpyHORecHitTag = cms.InputTag("reducedHcalRecHits:horeco")

process.ISpyMuon.iSpyMuonTag     = cms.InputTag("muons")
process.ISpyPhoton.iSpyPhotonTag = cms.InputTag("photons")
process.ISpyPFMET.iSpyPFMETTag   = cms.InputTag("pfMet")

# cmssw12 has a single jet tag (default iterativeCone5PFJets, absent in modern AOD);
# cmssw10/13/14 additionally have separate ak4/ak8 tags.
process.ISpyPFJet.iSpyPFJetTag = cms.InputTag("ak4PFJets")
if hasattr(process.ISpyPFJet, 'iSpyAK4PFJetTag'):
    process.ISpyPFJet.iSpyAK4PFJetTag = cms.InputTag("ak4PFJets")
    process.ISpyPFJet.iSpyAK8PFJetTag = cms.InputTag("ak8PFJets")

process.ISpyTrackExtrapolation.iSpyTrackExtrapolationTag            = cms.InputTag("trackExtrapolator")
process.ISpyTrackExtrapolation.iSpyGsfElectronTrackExtrapolationTag = cms.InputTag("gedGsfElectrons")
process.ISpyTrackExtrapolation.iSpyMuonTrackExtrapolationTag        = cms.InputTag("muons")
process.ISpyTrackExtrapolation.trackPtMin       = cms.double(options.trackPtMin)
process.ISpyTrackExtrapolation.electronPtMin    = cms.double(options.trackPtMin)
process.ISpyTrackExtrapolation.trackerMuonPtMin = cms.double(options.trackPtMin)

# ptMin only exists on branches that read it (cmssw10, cmssw14+)
if hasattr(process.ISpyMuon, 'ptMin'):
    process.ISpyMuon.ptMin = cms.double(options.muonPtMin)

process.ISpyVertex.iSpyPriVertexTag = cms.InputTag("offlinePrimaryVertices")
process.ISpyVertex.iSpySecVertexTag = cms.InputTag("inclusiveSecondaryVertices")

# ── Path ───────────────────────────────────────────────────────────────────

seq = getattr(process, ANALYZERS[0])
for name in ANALYZERS[1:]:
    seq = seq * getattr(process, name)
process.iSpy = cms.Path(seq)
process.schedule = cms.Schedule(process.iSpy)
