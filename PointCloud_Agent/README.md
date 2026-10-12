# PointCloud_Agent

Give it a CMS DAS dataset name; it produces, per event:

1. the **iSpy ig file** (same format the iSpy web display reads),
2. an **HTML visualization** of the event (plotly 3D, iSpy colours),
3. a **point cloud (`.npz`)** to use as machine-learning input.

```
/ZToMuMu_.../Run3Summer21DRPremix-120X_mcRun3_2021_realistic_v6-v2/AODSIM
        │  DAS: release, files        (lxplus host)
        ▼
  cmsRun + ISpy analyzers             (inside cmssw-el7 / el8 container)
        ▼
  NAME_0.ig … NAME_N.ig  ──► check_ig (are tracks/rechits really there?)
        ▼
  per event:  pointcloud/Run_X/Event_Y.npz   visualize/Run_X/Event_Y.html
```

Code is edited locally and pushed to GitHub; it is **run on lxplus**, and results stay on lxplus (EOS).

---

## 1. One-time setup on lxplus

```bash
ssh <cern-user>@lxplus.cern.ch

# a) get only this folder from the repository (sparse checkout)
git clone --filter=blob:none --sparse git@github.com:Jieun-H/CMS_iSpy_Analysis.git
cd CMS_iSpy_Analysis
git sparse-checkout set PointCloud_Agent

# b) python packages for the npz/html step (host python, not CMSSW's)
python3 -m pip install --user numpy plotly
```

- **Private repository?** First create an SSH key on lxplus (`ssh-keygen -t ed25519`) and add
  `~/.ssh/id_ed25519.pub` to GitHub → Settings → SSH and GPG keys.
- **Grid certificate**: `~/.globus/usercert.pem` and `userkey.pem` must exist and you must be in the CMS VO
  (https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookStartingGrid).

## 2. Every session

```bash
ssh <cern-user>@lxplus.cern.ch
cd ~/CMS_iSpy_Analysis && git pull          # get the latest code
voms-proxy-init --rfc --voms cms --valid 168:00
cd PointCloud_Agent
```

Run the agent from the **plain lxplus shell** — do not enter `cmssw-el7` yourself; the agent does it.

## 3. Run

```bash
python3 run_agent.py /ZToMuMu_M-120To200_TuneCP5_14TeV-powheg-pythia8/Run3Summer21DRPremix-120X_mcRun3_2021_realistic_v6-v2/AODSIM
```

Check what it will do first:
```bash
python3 run_agent.py <dataset> --dry-run
```

The first run for an era builds CMSSW + ISpy (10–20 min); later runs reuse it.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--nevents N` | 100 | events to process (`-1` = all events of the input files) |
| `--per-ig N` | 10 | events per ig file |
| `--nfiles N` | 1 | how many files of the DAS file list to read |
| `--files LFN ...` | – | use these files instead of the DAS list |
| `--name NAME` | `<primary>__<campaign>` | output folder / ig prefix |
| `--out DIR` | `/eos/user/<u>/<user>/PointCloud_Agent_output` | output base directory |
| `--cmssw-base DIR` | `/afs/cern.ch/work/<u>/<user>/pca_cmssw` | where CMSSW areas are created |
| `--globaltag GT` | from dataset name (MC) / era default (data) | override the GlobalTag |
| `--release CMSSW_X_Y_Z` | from DAS | override the release used to choose the era |
| `--no-pt-cut` | off | keep low-pT objects (ISpy default cuts: tracks 1 GeV, global muons 2 GeV) |
| `--steps` | `ig,pointcloud` | `pointcloud` alone re-makes npz/html from ig files already in the output |
| `--skip-content-check` | off | do not check the input collections before cmsRun |
| `--ig FILE ...` | – | convert existing ig files only (no CMSSW; also works on a laptop) |

## 4. Output

```
<out>/<name>/
  ig/          NAME_0.ig … NAME_N.ig, NAME_ES.ig (geometry), Events/Run_X/Event_Y (unzipped JSON)
  visualize/   Run_X/Event_Y.html
  pointcloud/  Run_X/Event_Y.npz
  logs/        setup_cmssw.log, eventcontent.txt, input_files.txt, cmsRun.log, check_ig.txt, manifest.json
```

`manifest.json` records the agent git commit, era, release, GlobalTag, input files and per-event object counts,
so every dataset can be traced back to the code that produced it.

View an HTML page: open it from CERNBox (https://cernbox.cern.ch → your EOS folder) or copy it to your laptop
(`scp <cern-user>@lxplus.cern.ch:<out>/<name>/visualize/Run_X/Event_Y.html .`).

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
| `hcal_hb` | x, y, z, energy, eta, phi, time, detid | box centre; E > 0 |

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
