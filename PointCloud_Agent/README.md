# PointCloud_Agent

Give it a CMS DAS dataset name; it produces, per event:

1. the **iSpy ig file** (same format the iSpy web display reads),
2. an **HTML visualization** of the event (plotly 3D, iSpy colours),
3. a **point cloud (`.npz`)** to use as machine-learning input.

```
/DYJetsToLL_.../RunIISummer20UL18RECO-106X_upgrade2018_realistic_v11_L1v1-v2/AODSIM
        │  DAS: release, files
        ▼
  cmsRun + ISpy analyzers  (CMSSW area this code lives in, e.g. CMSSW_10_6_30)
        ▼
  NAME_0.ig … NAME_N.ig  ──► check_ig (are tracks/rechits really there?)
        ▼
  per event:  pointcloud/Run_X/Event_Y_LSZ.npz   visualize/Run_X/Event_Y_LSZ.html
```

Code is edited locally and pushed to GitHub; it is **run on lxplus** from inside a CMSSW area,
and results stay on lxplus (EOS).

---

## 1. One-time setup on lxplus (Run2 UL: CMSSW_10_6_30)

The setup follows the **Quick start** of the iSpy analyzers README
(https://github.com/cms-outreach/ispy-analyzers/blob/cmssw10/README.md) step by step.
Three things differ from it:

| ispy-analyzers README | Here | Why |
|---|---|---|
| `CMSSW_10_2_9` | **`CMSSW_10_6_30`** | The README says to match the CMSSW version used to (re)process the data and to substitute your own version; Run2 UL was processed with CMSSW_10_6_X. |
| `git clone https://github.com/cms-outreach/ispy-analyzers.git ...` (no branch) | **`-b cmssw10`** | Without `-b` you get `master`, which has had no updates since 2022-03. |
| run directly on lxplus | **`cmssw-el7` first** | The README predates el9 lxplus; CMSSW_10_6_30 is an el7 (slc7) release. Its "lxplus with Singularity" section is still "COMING SOON". |

```bash
ssh <cern-user>@lxplus.cern.ch
cmssw-el7                                    # [added] enter the el7 container (lxplus is el9)
cd /afs/cern.ch/work/${USER:0:1}/$USER       # install location (or ~ ; AFS work has more quota)
export SCRAM_ARCH=slc7_amd64_gcc700          # [added] production architecture of CMSSW_10_6_30
```

Create a CMSSW environment:
```bash
cmsrel CMSSW_10_6_30
```

Change to the `CMSSW_10_6_30/src/` directory:
```bash
cd CMSSW_10_6_30/src/
```

Initialize the CMSSW environment:
```bash
cmsenv
```

Clone the necessary source code:
```bash
git clone -b cmssw10 https://github.com/cms-outreach/ispy-analyzers.git ISpy/Analyzers
git clone https://github.com/cms-outreach/ispy-services.git ISpy/Services
```

Compile the code:
```bash
scram b
```
The build ends with `>> Package ISpy/Analyzers built` (verified with the CMSSW_10_6_30 image).

Init your GRID proxy:
```bash
voms-proxy-init --rfc --voms cms
```

This is where the ispy-analyzers Quick start ends. Then, in the same `src/` directory, get this code and the one
python package the HTML step needs:
```bash
git clone git@github.com:Jieun-H/CMS_iSpy_Analysis.git   # small repo: code only
python3 -m pip install --user plotly                      # numpy comes with CMSSW
```

Resulting layout:
```
CMSSW_10_6_30/src/
  ISpy/Analyzers                       ispy-analyzers, branch cmssw10
  ISpy/Services                        ispy-services, master
  CMS_iSpy_Analysis/PointCloud_Agent   this code
```

- **Private repository?** First create an SSH key on lxplus (`ssh-keygen -t ed25519`) and add
  `~/.ssh/id_ed25519.pub` to GitHub → Settings → SSH and GPG keys.
- **Only `PointCloud_Agent/`?** A sparse checkout needs git ≥ 2.25, which the el7 container does not have.
  Run it from the plain lxplus (el9) shell in the same `src/` directory instead:
  `git clone --filter=blob:none --sparse git@github.com:Jieun-H/CMS_iSpy_Analysis.git && cd CMS_iSpy_Analysis && git sparse-checkout set PointCloud_Agent`
- **Grid certificate**: `~/.globus/usercert.pem` and `userkey.pem` must exist and you must be in the CMS VO
  (https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookStartingGrid).

## 2. Every session

```bash
ssh <cern-user>@lxplus.cern.ch
cmssw-el7
cd /afs/cern.ch/work/${USER:0:1}/$USER/CMSSW_10_6_30/src && cmsenv
voms-proxy-init --rfc --voms cms --valid 168:00
cd CMS_iSpy_Analysis && git pull origin main && cd PointCloud_Agent
```

## 3. Run

### (A) cmsRun only — make ig files by hand (recommended for the first test)

This is the step the ispy-analyzers README does with `cmsRun python/ispy_10_X_X_cfg.py` after editing the input
file in the cfg. Here the input and GlobalTag are passed as arguments instead of editing the cfg:

```bash
cd $CMSSW_BASE/src/CMS_iSpy_Analysis/PointCloud_Agent
dasgoclient -query="file dataset=<dataset>" | head -1          # pick one input file
cmsRun agent/cmssw/ispy_aod_cfg.py \
    inputFiles=<the /store/... path from above> \
    globalTag=<GlobalTag from the dataset name, e.g. 106X_upgrade2018_realistic_v11_L1v1> \
    outputDir=$PWD/test_ig outputName=test maxEvents=10
python3 agent/check_ig.py test_ig/test_0.ig                      # are tracks/rechits really in the ig?
```

As in the README, `test_0.ig` can be opened in the iSpy web display (http://cern.ch/ispy-webgl-dev).

### (B) The agent — ig + HTML + point cloud

```bash
python3 run_agent.py <dataset> --dry-run      # check the plan first
python3 run_agent.py <dataset>
```

- The agent uses the CMSSW area it lives in (`"cmssw_area_source": "enclosing"` in the plan) and the ISpy code
  cloned above; each run only does a quick `scram b`. If `ISpy/` is missing, the agent clones and builds it itself.
- If `ISpy/Analyzers` was cloned without `-b cmssw10`, the agent stops with
  `[!] ISpy/Analyzers is on branch 'master', expected 'cmssw10'`. Fix:
  `rm -rf $CMSSW_BASE/src/ISpy/Analyzers`, clone again with `-b cmssw10`, then `scram b`.
- The dataset must match the release: a Run3 dataset (`120X`) inside CMSSW_10_6_30 stops with a message.
  For Run3, repeat section 1 with `SCRAM_ARCH=slc7_amd64_gcc900`, `cmsrel CMSSW_12_0_4` and `-b cmssw12`.
- Running from the plain el9 shell also works: the agent then wraps CMSSW commands in `cmssw-el7` itself
  (python3 + `pip install --user numpy plotly` on the host).

### Options

| Option | Default | Meaning |
|---|---|---|
| `--nevents N` | 100 | events to process (`-1` = all events of the input files) |
| `--per-ig N` | 10 | events per ig file |
| `--nfiles N` | 1 | how many files of the DAS file list to read |
| `--first-file K` | 0 | skip the first K files of the DAS list (next batch of a dataset; give it a new `--name`) |
| `--no-html` | off | skip the per-event HTML pages (large productions for training) |
| `--files LFN ...` | – | use these files instead of the DAS list |
| `--name NAME` | `<primary>__<campaign>` | output folder / ig prefix |
| `--out DIR` | `/eos/user/<u>/<user>/PointCloud_Agent_output` | output base directory |
| `--cmssw-base DIR` | the CMSSW area this code is in | let the agent create/use `DIR/<release>` instead |
| `--globaltag GT` | from dataset name (MC) / era default (data) | override the GlobalTag |
| `--release CMSSW_X_Y_Z` | from DAS | override the release used to choose the era |
| `--no-pt-cut` | off | keep low-pT objects (ISpy default cuts: tracks 1 GeV, global muons 2 GeV) |
| `--steps` | `ig,pointcloud` | `pointcloud` alone re-makes npz/html from ig files already in the output |
| `--skip-content-check` | off | do not check the input collections before cmsRun |
| `--ig FILE ...` | – | convert existing ig files only (no CMSSW; also works on a laptop) |

## 4. Output

```
<out>/<name>/
  ig/          NAME_0.ig … NAME_N.ig, NAME_ES.ig (geometry), Events/Run_X/Event_Y_LSZ (unzipped JSON)
  visualize/   Run_X/Event_Y_LSZ.html
  pointcloud/  Run_X/Event_Y_LSZ.npz
  logs/        setup_cmssw.log, eventcontent.txt, input_files.txt, cmsRun.log, check_ig.txt, manifest.json
```

`manifest.json` records the agent git commit, era, release, GlobalTag, input files and per-event object counts,
so every dataset can be traced back to the code that produced it.

View an HTML page: open it from CERNBox (https://cernbox.cern.ch → your EOS folder) or copy it to your laptop
(`scp <cern-user>@lxplus.cern.ch:<out>/<name>/visualize/Run_X/Event_Y_LSZ.html .`).

### npz contents

```python
import numpy as np
d = np.load("Event_Y.npz")
d["tracks"]             # (N, 5)  x, y, z, pt, track_id    — 33 Bezier points per track (iSpy curve)
d["tracks_columns"]     # column names of the array above
```

| Key | Columns | Source |
|---|---|---|
| `tracks` | x, y, z, pt, track_id | `Tracks_V3` + `Extras_V1` (Bezier, 33 points) |
| `elec_tracks` | x, y, z, pt, track_id | `GsfElectrons_V2` (Bezier) |
| `muon_tracks` | x, y, z, pt, muon_id | `TrackerMuons_V2` (Bezier) |
| `global_muon` | x, y, z, pt, muon_id | `GlobalMuons_V1` → `Points_V1` |
| `standalone_muon` | x, y, z, pt, muon_id | `StandaloneMuons_V2` (Bezier) |
| `muon_chambers` | x, y, z, detid | `MuonChambers_V1`, box centre |
| `vertices` | x, y, z | `PrimaryVertices_V1` (Run2/3) or `Vertices_V1` (Run1) |
| `secondary_vertices` | x, y, z | `SecondaryVertices_V1` (Run2/3 only) |
| `ecal_eb`, `ecal_ee` | x, y, z, energy, eta, phi, time, detid | front-face diagonal intersection; E > 0.25 / 0.5 GeV |
| `hcal_hb`, `hcal_he`, `hcal_hf`, `hcal_ho` | x, y, z, energy, eta, phi, time, detid | box centre (8 corners); E > 0 |
| `track_info` | pt, eta, phi, charge, chi2, ndof, track_id | `Tracks_V3`, one row per track; row *i* = `track_id` *i* of `tracks` |
| `csc_segments` | x1, y1, z1, x2, y2, z2, endcap, station, ring, chamber, detid | `CSCSegments_V*`, both end points (`pos_1`, `pos_2`) |
| `dt_segments` | x1, y1, z1, x2, y2, z2, wheel, station, sector, detid | `DTRecSegment4D_V*`, both end points |
| `rpc_hits` | x, y, z, region, station, sector, layer, detid | `RPCRecHits_V*`, centre of the u/v/w cross |

Coordinates are in metres (iSpy convention). The arrays are identical to those of
`generate_pointcloud_260911.py` (checked on all 25 events of `DoubleElectron_Run2012C_0.ig`);
`has_gltf` was dropped and `secondary_vertices` / `*_columns` were added.

---

## Eras

The CMSSW release that produced the dataset (DAS: `dasgoclient -query="release dataset=<name>"`) decides the era.
ISpy's README asks to run in a release as close as possible to that one.

| Era | DAS release | CMSSW used | Container | ISpy branch | Status |
|---|---|---|---|---|---|
| `run2_ul` | `CMSSW_10_6_*` | CMSSW_10_6_30 | cmssw-el7 | `cmssw10` | not yet run on lxplus |
| `run3_120x` | `CMSSW_12_0_*`, `12_1_*` | CMSSW_12_0_4 | cmssw-el7 | `cmssw12` | not yet run on lxplus; branch CI builds CMSSW_12_0_0 |
| `run3_13x` | `CMSSW_12_4_*` – `13_*` | CMSSW_13_0_17 | cmssw-el8 | `cmssw13` | experimental |
| `run3_14x` | `CMSSW_14_*` | CMSSW_14_0_15 | (el9 host) | `cmssw14` | experimental |

Run1 (CMSSW_5_3_X) needs a different cfg; see `ig/Generate_ig/Run1_AOD/` in this repository.
Where the versions come from:
- ISpy branches: https://github.com/cms-outreach/ispy-analyzers/branches
- tested release per branch: `.gitlab-ci.yml` of the branch, e.g.
  https://github.com/cms-outreach/ispy-analyzers/blob/cmssw12/.gitlab-ci.yml
- release list / production architectures: https://cmssdt.cern.ch/SDT/releases.map

**GlobalTag**: for MC it is read from the dataset name (`…RECO-106X_upgrade2018_realistic_v11_L1v1-v2` →
`106X_upgrade2018_realistic_v11_L1v1`); for Run2 collision data `auto:run2_data`; for Run3 collision data pass
`--globaltag`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No valid grid proxy` | `voms-proxy-init --rfc --voms cms --valid 168:00` |
| `Run the agent from the plain lxplus shell` | you are inside `cmssw-elN`; `exit` and rerun |
| `Required collections missing` | input is not AOD, or a label differs: compare `logs/eventcontent.txt` with the InputTags in `agent/cmssw/ispy_aod_cfg.py` |
| `Input is MiniAOD` | use the AOD/AODSIM version of the dataset |
| xrootd errors in `cmsRun.log` | dataset on tape only (agent prints the sites) → choose another dataset or request a disk copy with Rucio |
| `ig files contain no tracks` | every collection was "not found" — see `logs/check_ig.txt` (ISpy error list) |
| `No data of type "GEMGeometry"` in ISpyMuon | cmssw10 ISpyMuon requires GEM geometry. `setup_cmssw.sh` patches `ISpyMuon.cc` (`getData` → `getHandle`); with a manual install apply the same `sed` (see the script) and `scram b` |
| `scram b` fails in `setup_cmssw.log` | ISpy branch does not compile in that release; report the log |
| `No module named plotly` | `python3 -m pip install --user numpy plotly` |

Why the old local Run2 ig files (`ig/ig_and_pointcloud/DYToMuMu_zmumu`) are empty: every event holds only
`Event_V3` + "are not found" errors. The input path in `ig/Generate_ig/Run2_AOD/generate_AOD_ig.py` does not
exist on eospublic, and `outputFilePath` was set (ISpyService also prepends it to the entry names inside the zip).
The agent checks the input collections before running and never sets `outputFilePath`.

## Layout

```
run_agent.py               entry point
agent/eras.py              era table, dataset name parsing, GlobalTag from name
agent/cmssw/setup_cmssw.sh cmsrel + clone ISpy + scram b          (runs in the container)
agent/cmssw/check_content.sh  edmDumpEventContent + required collections (container)
agent/cmssw/run_cmsrun.sh  cmsenv + cmsRun                        (container)
agent/cmssw/ispy_aod_cfg.py   cmsRun configuration (python2/3)
agent/check_ig.py          are the ig files non-empty?
agent/pointcloud.py        ig event → arrays → npz
agent/visualize.py         arrays → HTML
```
