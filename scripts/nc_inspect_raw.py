"""D-128 step 1: inventory of newly downloaded raw datasets (channels, sampling rate, trial and label counts).

No scores are computed; label counts are only tallied so the registration table (EOG/EMG channels, trials) can be fixed.
Usage: env/bin/python scripts/nc_inspect_raw.py configs/nc_raw.yaml
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("ESE_NUM_THREADS", "8"))
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io as sio
import yaml


def inspect_cho2017(root):
    out = {"subjects": {}}
    for f in sorted(Path(root).glob("s*.mat")):
        e = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["eeg"]
        n_ch = int(e.imagery_left.shape[0])
        out["subjects"][f.stem] = {
            "srate": float(e.srate), "n_channels": n_ch,
            "n_samples_left": int(e.imagery_left.shape[1]), "n_samples_right": int(e.imagery_right.shape[1]),
            "n_imagery_trials": int(e.n_imagery_trials),
            "n_bad_trials_voltage": [int(np.size(x)) for x in np.atleast_1d(e.bad_trial_indices.bad_trial_idx_voltage)]
            if hasattr(e, "bad_trial_indices") else None,
            "has_rest": hasattr(e, "rest"), "has_noise": hasattr(e, "noise"),
        }
        if "fields" not in out:
            out["fields"] = sorted(e._fieldnames)
    out["n_subjects"] = len(out["subjects"])
    out["note"] = "Channels 1-64 EEG (Biosemi 10-10), 65-68 EMG per dataset readme; verify against senloc/psenloc."
    return out


def inspect_bciciv2a(root):
    out = {"files": {}}
    for f in sorted(Path(root).glob("A0*.mat")):
        runs = np.atleast_1d(sio.loadmat(f, squeeze_me=True, struct_as_record=False)["data"])
        info = []
        for r in runs:
            y = np.atleast_1d(r.y)
            info.append({"n_samples": int(r.X.shape[0]), "n_channels": int(r.X.shape[1]), "fs": float(r.fs),
                         "n_trials": int(np.size(r.trial)), "labels": dict(Counter(int(v) for v in y)),
                         "n_artifact": int(np.sum(np.atleast_1d(r.artifacts))) if np.size(r.artifacts) else 0})
        out["files"][f.stem] = info
        if "classes" not in out:
            last = runs[-1]
            out["classes"] = [str(c) for c in np.atleast_1d(last.classes)]
    out["note"] = "BNCI 001-2014: 22 EEG + 3 EOG channels; trials per run 48 (runs with 0 trials are EOG/rest calibration)."
    return out


def inspect_physionet_mi(root):
    import mne
    mne.set_log_level("ERROR")
    imagery_runs = {4: "left_right_fist", 8: "left_right_fist", 12: "left_right_fist",
                    6: "fists_feet", 10: "fists_feet", 14: "fists_feet"}
    out = {"subjects": {}, "imagery_runs": imagery_runs}
    for sd in sorted(Path(root).glob("S[0-9][0-9][0-9]")):
        rec = {}
        for run, kind in imagery_runs.items():
            f = sd / f"{sd.name}R{run:02d}.edf"
            if not f.exists():
                rec[run] = None
                continue
            raw = mne.io.read_raw_edf(f, preload=False)
            desc = Counter(raw.annotations.description)
            rec[run] = {"sfreq": float(raw.info["sfreq"]), "n_channels": len(raw.ch_names),
                        "duration_s": float(raw.times[-1]), "annotations": dict(desc)}
            if "ch_names" not in out:
                out["ch_names"] = raw.ch_names
        out["subjects"][sd.name] = rec
    out["n_subjects"] = len(out["subjects"])
    sf = Counter(r["sfreq"] for s in out["subjects"].values() for r in s.values() if r)
    out["sfreq_counts"] = {str(k): v for k, v in sf.items()}
    return out


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    fns = {"cho2017": inspect_cho2017, "bciciv2a": inspect_bciciv2a, "physionet_mi": inspect_physionet_mi}
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, root in cfg["datasets"].items():
        res = fns[name](root)
        (out_dir / f"raw_inventory_{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(name, "done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
