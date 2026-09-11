import FWCore.ParameterSet.Config as cms

# Reference: https://github.com/cms-outreach/ispy-analyzers/blob/Run2012/produceIg.py
# (CMS 공식 Run2012 브랜치 예제 - 이 데이터셋과 동일한 시기/포맷용)
#
# 대상 데이터셋: https://opendata.cern.ch/record/6029
#   /DoubleElectron/Run2012C-22Jan2013-v1/AOD (실제 충돌 데이터, MC 아님)
#   CMSSW_5_3_32, GlobalTag: FT53_V21A_AN6
#
# 주의: 이 설정은 CMSSW_10_6_30이 아니라 CMSSW_5_3_32 + ISpy.Analyzers/ISpy.Services의
# "Run2012" 브랜치에서만 동작합니다 (ISpyVertex 등 analyzer의 파라미터 이름이
# CMSSW_10_6_30용 ISpy와 다릅니다). 컨테이너: cmsopendata/cmssw_5_3_32-slc6_amd64_gcc472

process = cms.Process("ISPY")

process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# 실제 데이터용 GlobalTag. Frontier(온라인 조건 DB) 대신
# CMS Open Data가 제공하는 cvmfs 로컬 스냅샷을 사용
process.GlobalTag.connect = cms.string('sqlite_file:/cvmfs/cms-opendata-conddb.cern.ch/FT53_V21A_AN6_FULL.db')
process.GlobalTag.globaltag = 'FT53_V21A_AN6::All'

process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring(
        'root://eospublic.cern.ch//eos/opendata/cms/Run2012C/DoubleElectron/AOD/22Jan2013-v1/20000/0032A48E-EA67-E211-AC23-0026189438BD.root'
    )
)

from FWCore.MessageLogger.MessageLogger_cfi import *
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 10

process.add_(
    cms.Service("ISpyService",
        outputFileName   = cms.untracked.string('DoubleElectron_Run2012C.ig'),
        outputESFileName = cms.untracked.string('DoubleElectron_Run2012C_ES.ig'),
        outputFilePath   = cms.untracked.string('/home/hju327/cms_open_data_work/opendata/ig/ig_files/'),
        outputMaxEvents  = cms.untracked.int32(10),
    )
)

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# ── Analyzers ──────────────────────────────────────────────────────────────

process.load('ISpy.Analyzers.ISpyEvent_cfi')

process.load('ISpy.Analyzers.ISpyEBRecHit_cfi')
process.load('ISpy.Analyzers.ISpyEERecHit_cfi')
process.load('ISpy.Analyzers.ISpyESRecHit_cfi')

process.load('ISpy.Analyzers.ISpyHBRecHit_cfi')
process.load('ISpy.Analyzers.ISpyHERecHit_cfi')
process.load('ISpy.Analyzers.ISpyHFRecHit_cfi')
process.load('ISpy.Analyzers.ISpyHORecHit_cfi')

process.load('ISpy.Analyzers.ISpyPFJet_cfi')
process.load('ISpy.Analyzers.ISpyPFMET_cfi')
process.load('ISpy.Analyzers.ISpyPFTau_cfi')

process.load('ISpy.Analyzers.ISpyMuon_cfi')
process.load('ISpy.Analyzers.ISpyPhoton_cfi')
process.load('ISpy.Analyzers.ISpySuperCluster_cfi')

process.load('ISpy.Analyzers.ISpyTrackExtrapolation_cfi')

process.load('ISpy.Analyzers.ISpyTriggerEvent_cfi')

process.load('ISpy.Analyzers.ISpyVertex_cfi')

# ── InputTags (2012 legacy AOD 컬렉션 이름 - Run2 UL과 다름) ─────────────────

process.ISpyEBRecHit.iSpyEBRecHitTag = cms.InputTag('reducedEcalRecHitsEB')
process.ISpyEERecHit.iSpyEERecHitTag = cms.InputTag('reducedEcalRecHitsEE')
process.ISpyESRecHit.iSpyESRecHitTag = cms.InputTag('reducedEcalRecHitsES')

process.ISpyHBRecHit.iSpyHBRecHitTag = cms.InputTag('reducedHcalRecHits:hbhereco')
process.ISpyHERecHit.iSpyHERecHitTag = cms.InputTag('reducedHcalRecHits:hbhereco')
process.ISpyHFRecHit.iSpyHFRecHitTag = cms.InputTag('reducedHcalRecHits:hfreco')
process.ISpyHORecHit.iSpyHORecHitTag = cms.InputTag('reducedHcalRecHits:horeco')

process.ISpyMuon.iSpyMuonTag = cms.InputTag('muons')

process.ISpyPFJet.iSpyPFJetTag = cms.InputTag('ak7PFJets')
process.ISpyPFMET.iSpyPFMETTag = cms.InputTag('pfMet')
process.ISpyPFTau.iSpyPFTauTag = cms.InputTag('hpsPFTauProducer')

process.ISpyPhoton.iSpyPhotonTag = cms.InputTag('photons')

process.ISpySuperCluster.iSpySuperClusterTag = cms.InputTag('correctedHybridSuperClusters')

process.ISpyTriggerEvent.triggerEventTag = cms.InputTag('hltTriggerSummaryAOD')
process.ISpyTriggerEvent.triggerResultsTag = cms.InputTag('TriggerResults')
process.ISpyTriggerEvent.processName = cms.string('HLT')

# 주의: 이 시기 ISpyVertex는 iSpyVertexTag 단일 파라미터만 사용 (Pri/Sec 구분 없음)
process.ISpyVertex.iSpyVertexTag = cms.InputTag('offlinePrimaryVertices')

# ── Path ───────────────────────────────────────────────────────────────────

process.iSpy = cms.Path(
    process.ISpyEvent *
    process.ISpyEBRecHit *
    process.ISpyEERecHit *
    process.ISpyESRecHit *
    process.ISpyHBRecHit *
    process.ISpyHERecHit *
    process.ISpyHFRecHit *
    process.ISpyHORecHit *
    process.ISpyMuon *
    process.ISpyPFJet *
    process.ISpyPFMET *
    process.ISpyPFTau *
    process.ISpyPhoton *
    process.ISpySuperCluster *
    process.ISpyTrackExtrapolation *
    process.ISpyTriggerEvent *
    process.ISpyVertex
)

process.schedule = cms.Schedule(process.iSpy)
