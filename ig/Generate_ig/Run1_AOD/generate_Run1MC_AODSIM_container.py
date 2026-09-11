# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms

# 2012(Run1) 시절 MC AODSIM -> ig 변환용 cfg.
# generate_DoubleElectron_Run2012C_container.py (실제 데이터, 검증 완료)를 기반으로
# MC용으로 바꾼 버전. 이 파일 자체는 그대로 두고, 실행 전 아래 두 곳
# ">>> 여기를 채워주세요 <<<" 표시된 부분만 채우면 됩니다.
#
# 실행 조건 (실제 데이터 변환 때와 동일):
#   - CMSSW_5_3_32 + ISpy.Analyzers/ISpy.Services의 "Run2012" 브랜치
#   - 컨테이너: cmsopendata/cmssw_5_3_32 (--cap-add SYS_ADMIN --device /dev/fuse 로 생성,
#     컨테이너 안에 cvmfs 네이티브 마운트 필요 - PROCESS_LOG.md 부록 참고)
#   - GlobalTag DB가 상대경로로 하위 파일을 참조하므로, cmsRun은 반드시
#     /cvmfs/cms-opendata-conddb.cern.ch 안에서 실행:
#       cd /cvmfs/cms-opendata-conddb.cern.ch && cmsRun .../generate_Run1MC_AODSIM_container.py

process = cms.Process("ISPY")

process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# ============================================================================
# >>> 여기를 채워주세요 (REQUIRED) 1/2: MC GlobalTag <<<
# ----------------------------------------------------------------------------
# 실제 데이터용 GT(FT53_V21A_AN6)가 아니라 MC 생산에 쓰인 GT를 써야 합니다.
# 보통 opendata.cern.ch의 데이터셋 경로 이름 안에 GT가 들어있습니다, 예:
#   /.../Summer12_DR53X-PU_S10_START53_V7A-v1/AODSIM  ->  START53_V7A 계열
# 아래는 이 cvmfs 저장소(/cvmfs/cms-opendata-conddb.cern.ch/)에서 실제로 확인한
# Run1 MC용 GlobalTag 목록입니다. 본인 데이터셋 이름과 가장 가까운 것으로 바꾸세요:
#   START53_V7N   START53_V27   START53_LV6A1   START42_V17B
#   (중이온 MC라면: STARTHI53_LV1 / STARTHI53_V27 / STARTHI53_V28)
GC_MC_GLOBALTAG = 'START53_V7N'   # <-- 이 문자열을 본인 데이터셋에 맞는 태그로 교체
process.GlobalTag.connect = cms.string(
    'sqlite_file:/cvmfs/cms-opendata-conddb.cern.ch/%s.db' % GC_MC_GLOBALTAG
)
process.GlobalTag.globaltag = '%s::All' % GC_MC_GLOBALTAG
# ============================================================================

# ============================================================================
# >>> 여기를 채워주세요 (REQUIRED) 2/2: 변환할 AODSIM 파일 경로 <<<
# ----------------------------------------------------------------------------
# opendata.cern.ch에서 받은 AODSIM 루트 파일의 xrootd 경로(root://eospublic.cern.ch/...)
# 또는 로컬에 이미 내려받은 파일이면 그 절대경로를 그대로 넣으면 됩니다.
process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring(
        'root://eospublic.cern.ch//eos/opendata/cms/.../AODSIM 파일 경로를 여기에.root'
    )
)
# ============================================================================

from FWCore.MessageLogger.MessageLogger_cfi import *
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 10

process.add_(
    cms.Service("ISpyService",
        # outputFilePath는 이 ISpy 버전에서 무시되므로(PROCESS_LOG.md 5번 참고),
        # outputFileName/outputESFileName에 절대경로로 직접 지정.
        outputFileName   = cms.untracked.string('/home/cmsusr/work/opendata/ig/ig_files/Run1MC_verify/Run1MC_AODSIM.ig'),
        outputESFileName = cms.untracked.string('/home/cmsusr/work/opendata/ig/ig_files/Run1MC_verify/Run1MC_AODSIM_ES.ig'),
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

# ── InputTags ────────────────────────────────────────────────────────────
# AODSIM은 실제 데이터 AOD와 RECO 레벨 컬렉션 이름이 동일합니다 (같은 재구성
# 소프트웨어를 거치기 때문). 아래는 검증된 실제 데이터 cfg와 동일하게 둔 것이며,
# 보통 그대로 쓰면 됩니다. 컬렉션이 없다는 에러가 나면 해당 줄만 데이터셋에 맞게 수정하세요.

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

process.ISpyVertex.iSpyVertexTag = cms.InputTag('offlinePrimaryVertices')

# 참고: MC 진짜 입자 정보(생성 레벨)까지 보고 싶다면 ISpy.Analyzers 안에
# ISpyGenJet_cfi / ISpySimTrack_cfi 등을 추가로 load해서 Path에 넣으면 됩니다
# (이 cfg에는 기본 비활성 상태).

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
