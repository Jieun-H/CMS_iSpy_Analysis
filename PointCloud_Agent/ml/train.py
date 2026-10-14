"""
Train ParticleNet (ml/particlenet.py) on a pack made by ml/pack.py.

Every choice is a command-line option (nothing to edit): task, track encoder, track
features, calorimeter parts, muon-system inputs, label pt_min / match_tol, model size,
optimiser, train fraction (learning curve). Each run writes to its own --out folder:

    config.json          all options, input config, normalisation, pack info, git commit
    metrics.jsonl        one line per epoch (train loss, val loss / AUC / AP, time)
    last.pt, best.pt     checkpoints: weights, optimiser + scheduler state, everything above
    test_metrics.json    best.pt on the test split
    test_predictions.npz per-track (or per-event) score, label, pt, event index

    # muon vs non-muon per track, hit_keys setting B (HANDOFF section 9), on GPU
    python3 -m ml.train --pack /eos/user/j/jieun/ml/packed/zmm_ul16_v1 --out RUN_DIR
    # ablations
    python3 -m ml.train --pack PACK --out RUN --track-encoder mlp
    python3 -m ml.train --pack PACK --out RUN --hit-keys ecal_eb ecal_ee
    python3 -m ml.train --pack PACK --out RUN --no-segments --no-rpc
    python3 -m ml.train --pack PACK --out RUN --train-fraction 0.1
    python3 -m ml.train --pack PACK --out RUN --task event        # has_muon per event
    # continue a stopped run (same --out)
    python3 -m ml.train --pack PACK --out RUN --resume
    # quick check that everything runs (CPU, few batches)
    python3 -m ml.train --pack PACK --out /tmp/t --epochs 1 --limit-batches 3 --batch-size 4
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml import inputs, packed  # noqa: E402
from ml.metrics import summarise  # noqa: E402
from ml.pack import git_info  # noqa: E402
from ml.particlenet import TrackParticleNet  # noqa: E402


# ---------------------------------------------------------------- data

class EventDataset(torch.utils.data.Dataset):
    def __init__(self, split, cfg, norm, indices=None):
        self.split, self.cfg, self.norm = split, cfg, norm
        self.indices = np.arange(len(split)) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        e = int(self.indices[i])
        out = packed.event_inputs(self.split.event(e), self.cfg, self.norm)
        out["index"] = e
        return out


def collate(events):
    b = packed.pad_batch(events)
    b = {k: torch.from_numpy(v) for k, v in b.items()}
    b["index"] = torch.tensor([e["index"] for e in events])
    return b


def to_device(b, device):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


# ---------------------------------------------------------------- loop

def run_epoch(model, loader, device, task, pos_weight, optimiser=None, scaler=None,
              amp=False, limit=None, collect=False):
    train = optimiser is not None
    model.train(train)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum",
                                         pos_weight=torch.tensor(pos_weight, device=device))
    total, count = 0.0, 0
    scores, labels, pts, etas, events = [], [], [], [], []
    for step, b in enumerate(loader):
        if limit is not None and step >= limit:
            break
        b = to_device(b, device)
        with torch.set_grad_enabled(train), \
                torch.autocast(device_type=device.type, enabled=amp):
            logit = model(b)
        logit = logit.float()
        if task == "track":
            m = b["track_mask"]
            lg, lb = logit[m], b["track_label"][m]
        else:
            lg, lb = logit, b["has_muon"]
        loss = loss_fn(lg, lb)
        n = max(len(lb), 1)
        if train:
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss / n).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimiser)
            scaler.update()
        total += float(loss.detach())
        count += len(lb)
        if collect:
            scores.append(torch.sigmoid(lg).detach().cpu().numpy())
            labels.append(lb.cpu().numpy())
            if task == "track":
                pts.append(b["track_pt"][m].cpu().numpy())
                etas.append(b["track_eta"][m].cpu().numpy())
                events.append(b["index"].unsqueeze(1).expand_as(m)[m].cpu().numpy())
            else:
                events.append(b["index"].cpu().numpy())
    out = {"loss": total / max(count, 1)}
    if collect:
        cat = (lambda x: np.concatenate(x) if x else np.zeros(0))
        out.update(score=cat(scores), label=cat(labels), pt=cat(pts), eta=cat(etas),
                   event=cat(events))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--pack", required=True, help="folder made by ml/pack.py")
    g.add_argument("--out", required=True, help="run folder (created; one per experiment)")
    g.add_argument("--task", choices=("track", "event"), default="track",
                   help="track: is this track a muon (default); event: has_muon")
    g.add_argument("--track-features", nargs="+", default=list(inputs.TRACK_INFO_FEATURES),
                   choices=inputs.TRACK_INFO_FEATURES, help="per-track scalars (default all)")
    g.add_argument("--hit-keys", nargs="*", default=["ecal_eb", "ecal_ee", "hcal_hb", "hcal_he"],
                   choices=inputs.HIT_KEYS, help="calorimeter parts (default: setting B)")
    g.add_argument("--segments", action=argparse.BooleanOptionalAction, default=True,
                   help="CSC/DT segments as nodes (default on)")
    g.add_argument("--rpc", action=argparse.BooleanOptionalAction, default=True,
                   help="RPC hits as nodes (default on)")
    g.add_argument("--max-hits", type=int, default=None, help="keep only the N highest-energy hits")
    g.add_argument("--pt-min", type=float, default=10.0, help="muon pT for the labels (default 10)")
    g.add_argument("--match-tol", type=float, default=inputs.MATCH_TOL,
                   help=f"track <-> tracker muon distance in m (default {inputs.MATCH_TOL:g})")
    g.add_argument("--train-fraction", type=float, default=1.0, help="use this fraction of train")
    g = p.add_argument_group("model")
    g.add_argument("--track-encoder", choices=("conv", "mlp"), default="conv")
    g.add_argument("--dim", type=int, default=64, help="node embedding width")
    g.add_argument("--k", type=int, default=16, help="neighbours per node")
    g.add_argument("--edge-channels", default="64,64,64 128,128,128 256,256,256",
                   help="EdgeConv blocks, space-separated, channels comma-separated")
    g.add_argument("--fusion", type=int, default=128)
    g.add_argument("--dropout", type=float, default=0.1)
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--pos-weight", default="1",
                   help="BCE weight of the positive class, a number or 'auto' (= neg/pos in train)")
    g.add_argument("--patience", type=int, default=None, help="stop after N epochs without val gain")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    g.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1 ...")
    g.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="mixed precision on CUDA (default on)")
    g.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    g.add_argument("--limit-batches", type=int, default=None, help="batches per epoch (quick tests)")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    amp = args.amp and device.type == "cuda"
    os.makedirs(args.out, exist_ok=True)
    log = (lambda msg: print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True))
    log(f"device {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
        + f", torch {torch.__version__}, amp {amp}")

    with open(os.path.join(args.pack, "pack.json")) as f:
        pack_info = json.load(f)
    cfg = packed.InputConfig(args.track_features, args.hit_keys, args.segments, args.rpc,
                             args.pt_min, args.match_tol, args.max_hits)
    norm = packed.Normalizer(pack_info["stats"], cfg)
    splits = {s: packed.PackedSplit(os.path.join(args.pack, s)) for s in ("train", "val", "test")}

    train_idx = np.arange(len(splits["train"]))
    if args.train_fraction < 1:
        rng = np.random.default_rng(args.seed)
        train_idx = np.sort(rng.permutation(train_idx)[:max(1, int(args.train_fraction * len(train_idx)))])

    if args.pos_weight == "auto":
        tr = splits["train"]
        if args.task == "track":
            lab = (np.asarray(tr.array("track_mu_dist")) < cfg.match_tol) & \
                  (np.asarray(tr.array("track_mu_pt")) > cfg.pt_min)
        else:
            lab = (np.asarray(tr.array("glb_pt")) > cfg.pt_min).any(1)
        pos_weight = float((~lab).sum() / max(lab.sum(), 1))
    else:
        pos_weight = float(args.pos_weight)

    loader_kw = dict(batch_size=args.batch_size, collate_fn=collate, num_workers=args.workers,
                     pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    loaders = {
        "train": torch.utils.data.DataLoader(EventDataset(splits["train"], cfg, norm, train_idx),
                                             shuffle=True, drop_last=True, **loader_kw),
        "val": torch.utils.data.DataLoader(EventDataset(splits["val"], cfg, norm), **loader_kw),
        "test": torch.utils.data.DataLoader(EventDataset(splits["test"], cfg, norm), **loader_kw),
    }

    model_kw = {
        "n_scalars": len(packed.track_scalar_names(cfg.track_features)),
        "n_types": len(cfg.node_types), "other_width": packed.OTHER_WIDTH,
        "task": args.task, "track_encoder": args.track_encoder, "dim": args.dim, "k": args.k,
        "edge_channels": [[int(c) for c in blk.split(",")] for blk in args.edge_channels.split()],
        "fusion": args.fusion, "dropout": args.dropout,
    }
    model = TrackParticleNet(**model_kw).to(device)
    n_param = sum(q.numel() for q in model.parameters())
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
    except (AttributeError, TypeError):          # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    config = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": os.uname().nodename, "command": sys.argv, "args": vars(args),
        "git": git_info(), "device": str(device), "torch": torch.__version__,
        "input_config": cfg.to_json(), "node_types": cfg.node_types,
        "track_scalars": packed.track_scalar_names(cfg.track_features),
        "normalisation": norm.to_json(), "model": model_kw, "n_parameters": n_param,
        "pos_weight": pos_weight, "train_events": int(len(train_idx)),
        "pack": {"path": os.path.abspath(args.pack), "git": pack_info.get("git"),
                 "created": pack_info.get("created"), "summary": pack_info.get("summary"),
                 "split_seed": pack_info.get("seed")},
    }
    split_lists = {"train": [splits["train"].paths[i] for i in train_idx],
                   "val": splits["val"].paths, "test": splits["test"].paths}

    history, start_epoch, best, stale = [], 0, -np.inf, 0
    last_path, best_path = os.path.join(args.out, "last.pt"), os.path.join(args.out, "best.pt")
    if args.resume:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimiser.load_state_dict(ck["optimiser"])
        scheduler.load_state_dict(ck["scheduler"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        history, start_epoch, best, stale = ck["history"], ck["epoch"] + 1, ck["best"], ck["stale"]
        log(f"resumed from {last_path} at epoch {start_epoch}")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    log(f"train {len(train_idx)} / val {len(splits['val'])} / test {len(splits['test'])} events, "
        f"{n_param:,} parameters, pos_weight {pos_weight:.3g}, nodes {cfg.node_types}")

    key = "auc_pt" if args.task == "track" else "auc"

    def save(path, epoch):
        torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best, "stale": stale, "history": history,
                    "config": config, "splits": split_lists}, path + ".tmp")
        os.replace(path + ".tmp", path)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr = run_epoch(model, loaders["train"], device, args.task, pos_weight, optimiser, scaler,
                       amp, args.limit_batches)
        va = run_epoch(model, loaders["val"], device, args.task, pos_weight, amp=amp,
                       limit=args.limit_batches, collect=True)
        scheduler.step()
        m = summarise(va["score"], va["label"], va["pt"], cfg.pt_min, args.task)
        row = {"epoch": epoch, "lr": optimiser.param_groups[0]["lr"], "train_loss": tr["loss"],
               "val_loss": va["loss"], **{f"val_{k}": v for k, v in m.items()},
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        with open(os.path.join(args.out, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        score = m.get(key) if m.get(key) is not None else -va["loss"]
        improved = score > best
        if improved:
            best, stale = score, 0
        else:
            stale += 1
        save(last_path, epoch)
        if improved:
            save(best_path, epoch)
        log(f"epoch {epoch}: train loss {tr['loss']:.4f}  val loss {va['loss']:.4f}  "
            f"val {key} {m.get(key)}  ap {m.get('ap_pt' if args.task == 'track' else 'ap')}  "
            f"{row['seconds']}s" + ("  *best" if improved else ""))
        if args.patience is not None and stale >= args.patience:
            log(f"no val gain for {stale} epochs, stopping")
            break

    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    te = run_epoch(model, loaders["test"], device, args.task, pos_weight, amp=amp,
                   limit=args.limit_batches, collect=True)
    m = summarise(te["score"], te["label"], te["pt"], cfg.pt_min, args.task)
    m.update({"loss": te["loss"], "best_epoch": ck["epoch"]})
    with open(os.path.join(args.out, "test_metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    np.savez(os.path.join(args.out, "test_predictions.npz"), score=te["score"], label=te["label"],
             pt=te["pt"], eta=te["eta"], event=te["event"])
    log(f"test (best epoch {ck['epoch']}): {json.dumps(m)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
