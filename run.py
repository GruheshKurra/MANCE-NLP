"""Train and evaluate MANCE on AG News, DBpedia and GermEval 2014.

    python run.py train configs/ag_news.yaml --seed 1
    python run.py train configs/ag_news.yaml --seed 1 --tag no_char_order --set model.char_positions=false
    python run.py table

A run trains on the train split, keeps the epoch with the best validation score, and evaluates
that checkpoint once on the test split (and on the unseen-variant test sets, if configured).
Each run writes experiments/runs/<dataset>_<tag>_seed<seed>.json.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from mance import data
from mance.model import Model
from mance.oov import lowercase_counts, perturb

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "experiments", "runs")
CHECKPOINTS = os.path.join(ROOT, "experiments", "checkpoints")
CACHE = os.path.join(ROOT, ".cache")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_for(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def override(cfg, assignments):
    for item in assignments or []:
        key, value = item.split("=", 1)
        *path, last = key.split(".")
        node = cfg
        for part in path:
            node = node[part]
        node[last] = yaml.safe_load(value)


def git_state():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def build_data(cfg, seed):
    d, oov = cfg["data"], cfg.get("oov") or {}
    key = json.dumps([cfg["dataset"], d, oov, seed], sort_keys=True)
    path = os.path.join(CACHE, f"{cfg['dataset']}-{hashlib.sha1(key.encode()).hexdigest()[:10]}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)

    splits, label_names = data.load_splits(cfg["dataset"])
    splits = data.hold_out(splits, d["val_fraction"], seed)
    if d.get("max_train"):
        sentences, labels = splits["train"]
        keep = np.random.default_rng(seed).permutation(len(sentences))[: d["max_train"]]
        splits["train"] = ([sentences[i] for i in keep], [labels[i] for i in keep])

    vocab = data.Vocab(splits["train"][0], d["word_min_freq"], d["vocab_size"])
    encode = lambda sentences, labels: data.encode(
        sentences, labels, cfg["task"], vocab, d["max_words"], d["max_chars"])
    out = {name: encode(*splits[name]) for name in ("train", "validation", "test")}
    out["sizes"] = {name: len(splits[name][0]) for name in ("train", "validation", "test")}
    out["label_names"] = label_names
    out["vocab_size"] = len(vocab)

    if oov.get("fraction"):
        lexicon = lowercase_counts(data.load_splits(oov["lexicon"])[0]["train"][0])
        test_sentences, test_labels = splits["test"]
        out["oov"] = {}
        for name, lex in (("rule", None), ("attested", lexicon)):
            changed, stats = perturb(test_sentences, vocab.seen, oov["fraction"], oov["seed"],
                                     d["max_words"], lex, oov["min_count"])
            stats["examples"] = [(a, b) for old, new in zip(test_sentences, changed)
                                 for a, b in zip(old, new) if a != b][:20]
            out["oov"][name] = {"data": encode(changed, test_labels), "stats": stats}

    os.makedirs(CACHE, exist_ok=True)
    torch.save(out, path)
    return out


def batches(split, size, device, shuffle=False, generator=None):
    """Yields (uniq, slots, word_ids, mask, labels).

    uniq holds each distinct word of the batch once, padded to a multiple of 128 rows so the
    number of distinct tensor shapes stays small; slots maps every word position to its row.
    """
    n = len(split["chars"])
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, size):
        idx = order[start : start + size]
        chars = split["chars"][idx]
        width = int((chars[:, :, 0] > 0).any(0).nonzero().max()) + 1
        width = -(-width // 8) * 8
        chars = chars[:, :width]
        uniq, slots = np.unique(chars.reshape(-1, chars.size(-1)).numpy(), axis=0, return_inverse=True)
        filler = np.zeros((-len(uniq) % 128, uniq.shape[1]), dtype=uniq.dtype)
        filler[:, 0] = data.UNK  # an all-padding row would make attention undefined
        labels = split["labels"][idx]
        if labels.dim() == 2:
            labels = labels[:, :width]
        yield (torch.from_numpy(np.concatenate([uniq, filler])).long().to(device),
               torch.from_numpy(slots.reshape(chars.shape[:2])).to(device),
               split["words"][idx, :width].to(device),
               (chars[:, :, 0] > 0).to(device),
               labels.to(device))


def autocast(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def loss_fn(logits, labels):
    return F.cross_entropy(logits.flatten(0, -2), labels.flatten(), ignore_index=-100)


@torch.no_grad()
def evaluate(model, split, cfg, device, label_names):
    model.eval()
    if cfg["task"] == "cls":
        correct = 0
        for *inputs, labels in batches(split, cfg["train"]["eval_batch_size"], device):
            with autocast(device):
                correct += (model(*inputs).argmax(-1) == labels).sum().item()
        return {"accuracy": 100 * correct / len(split["labels"])}

    from seqeval.metrics import f1_score, precision_score, recall_score

    gold, pred = [], []
    for *inputs, labels in batches(split, cfg["train"]["eval_batch_size"], device):
        with autocast(device):
            guesses = model(*inputs).argmax(-1)
        for g_row, p_row in zip(labels.tolist(), guesses.tolist()):
            pairs = [(g, p) for g, p in zip(g_row, p_row) if g != -100]
            gold.append([label_names[g] for g, _ in pairs])
            pred.append([label_names[p] for _, p in pairs])
    return {"f1": 100 * f1_score(gold, pred), "precision": 100 * precision_score(gold, pred),
            "recall": 100 * recall_score(gold, pred)}


def train(args):
    cfg = yaml.safe_load(open(args.config))
    override(cfg, args.set)
    if args.max_train:
        cfg["data"]["max_train"] = args.max_train
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    tag, metric = args.tag or "mance", "accuracy" if cfg["task"] == "cls" else "f1"
    device = device_for(args.device)
    seed_everything(args.seed)
    started = time.time()

    ds = build_data(cfg, args.seed)
    print(f"{cfg['dataset']}/{tag} seed {args.seed} on {device}: {ds['sizes']}", flush=True)

    seed_everything(args.seed)
    model = Model(cfg, len(ds["label_names"]), ds["vocab_size"], data.NUM_CHARS).to(device)
    t = cfg["train"]
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    warmup = t["warmup_steps"]
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, (step + 1) / warmup))
    generator = torch.Generator().manual_seed(args.seed)

    best, best_state, stale, history = float("-inf"), None, 0, []
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        tick, total = time.time(), 0.0
        for step, (*inputs, labels) in enumerate(batches(ds["train"], t["batch_size"], device, True, generator), 1):
            with autocast(device):
                loss = loss_fn(model(*inputs), labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            schedule.step()
            total += loss.item()
            if step % 500 == 0:
                print(f"  epoch {epoch} step {step} loss {total / step:.4f} ({time.time() - tick:.0f}s)", flush=True)
        val = evaluate(model, ds["validation"], cfg, device, ds["label_names"])
        history.append({"epoch": epoch, "loss": total / step, "val": val, "seconds": round(time.time() - tick)})
        print(f"epoch {epoch}: loss {total / step:.4f}, val {metric} {val[metric]:.2f} ({time.time() - tick:.0f}s)", flush=True)
        if val[metric] > best:
            best, stale = val[metric], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale == t["patience"]:
                break

    model.load_state_dict(best_state)
    name = f"{cfg['dataset']}_{tag}_seed{args.seed}"
    os.makedirs(CHECKPOINTS, exist_ok=True)
    torch.save(best_state, os.path.join(CHECKPOINTS, name + ".pt"))

    result = {"test": evaluate(model, ds["test"], cfg, device, ds["label_names"])}
    for kind, perturbed in ds.get("oov", {}).items():
        score = evaluate(model, perturbed["data"], cfg, device, ds["label_names"])
        result[f"oov_{kind}"] = {"test": score, "drop": result["test"][metric] - score[metric],
                                 "stats": perturbed["stats"]}
    print(f"test {metric} {result['test'][metric]:.2f}", flush=True)

    record = {
        "dataset": cfg["dataset"], "tag": tag, "seed": args.seed, "metric": metric,
        "result": result, "best_epoch": max(history, key=lambda h: h["val"][metric])["epoch"],
        "history": history, "params": sum(p.numel() for p in model.parameters()),
        "sizes": ds["sizes"], "config": cfg,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else str(device),
        "torch": torch.__version__, "python": platform.python_version(), "git": git_state(),
        "dataset_revision": data.dataset_revision(cfg["dataset"]),
        "minutes": round((time.time() - started) / 60, 1),
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, name + ".json"), "w") as f:
        json.dump(record, f, indent=1)


def table(_):
    runs = {}
    for path in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        r = json.load(open(path))
        runs.setdefault((r["dataset"], r["tag"]), []).append(r)

    def summary(values):
        if not values:
            return "–"
        if len(values) == 1:
            return f"{values[0]:.2f}"
        return f"{np.mean(values):.2f} ± {np.std(values, ddof=1):.2f}"

    lines = [
        "| Dataset | Model | Test score | Seeds | Drop with unseen variants: rule set | attested set | Params |",
        "|---|---|---|---|---|---|---|",
    ]
    for (dataset, tag), rs in sorted(runs.items()):
        scores = [r["result"]["test"][r["metric"]] for r in rs]
        cells = []
        for kind in ("rule", "attested"):
            key = f"oov_{kind}"
            drops = [r["result"][key]["drop"] for r in rs if key in r["result"]]
            share = next((r["result"][key]["stats"]["fraction"] for r in rs if key in r["result"]), None)
            cells.append(summary(drops) + (f" ({100 * share:.1f}% of words)" if share else ""))
        lines.append(f"| {dataset} | {tag} | {summary(scores)} | {len(rs)} | {cells[0]} | {cells[1]} "
                     f"| {rs[0]['params'] / 1e6:.1f}M |")

    with open(os.path.join(ROOT, "experiments", "RESULTS.md"), "w") as f:
        f.write("# Results\n\nAccuracy (%) for AG News and DBpedia, entity-level F1 for GermEval 2014. "
                "Measured values are mean ± sd over seeds. Built by `python run.py table` "
                "from `experiments/runs/`.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train")
    t.add_argument("config")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--tag")
    t.add_argument("--set", action="append", metavar="KEY=VALUE")
    t.add_argument("--device", default="auto")
    t.add_argument("--max-train", type=int)
    t.add_argument("--epochs", type=int)
    commands.add_parser("table")
    args = parser.parse_args()
    {"train": train, "table": table}[args.command](args)


if __name__ == "__main__":
    main()
