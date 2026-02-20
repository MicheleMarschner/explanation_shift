"""
Output: pairs_table_from_pt.csv

This step creates one big per-pair dataset across all corruptions and severities. Each row is a (clean, corrupted) 
pair with labels, correctness, entropy change, slice masks (invariant / both_correct), and explanation-similarity 
values (cos / IoU / rho). It’s the common table you need so all later plots can be computed consistently.


Plot C (pooled scatter): plotC_pooled_and_grouped(... grouped=False)

This plot shows whether prediction uncertainty shift (|ΔEntropy|) and explanation drift tend to increase together 
across all conditions. Each point is one image pair (optionally restricted to a slice like invariant), so you can 
see whether large entropy changes correspond to large explanation drift (and the Spearman ρ summarizes that trend).


Plot C (grouped scatter): plotC_pooled_and_grouped(... grouped=True)

This produces the same scatter plot separately for each corruption × severity. It shows which corruption settings 
have a strong coupling between |ΔEntropy| and explanation drift, and which settings show decoupling (e.g., 
explanations drift a lot even when entropy change is small, or vice versa).


Spearman table: spearman_table(...)

This table summarizes the relationship between |ΔEntropy| and explanation drift as a single number (Spearman ρ) 
for each corruption × severity, plus a pooled row per corruption. It’s the “compressed” version of Plot C for 
reporting and comparisons.


Violin plot: violin_by_severity(...)

This plot shows how the distribution of explanation stability (e.g., rho) changes with severity. Instead of one 
mean per severity, it shows the full spread (median + variability), so you can see whether severity increases 
cause (a) a uniform shift, (b) heavier tails, or (c) only some samples becoming unstable.

"""


from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.configs.global_config import PATHS


# ---------- CONFIG: adapt these to your saved vector key names ----------
# Your vectors dict contains exp vectors under prefix_keys(exp_vec_all, "exp__")
# so keys might be: "exp__cos", "exp__iou", "exp__rho" OR "exp__cos_vec", etc.
COS_KEY_CANDIDATES = ["exp__cosine_sim"]
IOU_KEY_CANDIDATES = ["exp__iou_topk"]
RHO_KEY_CANDIDATES = ["exp__spearman_rho"]

# Masks / entropy keys you said you have
DH_KEY = "dH"
INV_KEY = "invariant"
BC_KEY  = "both_correct"

PRED_CORR_KEYS = ["pred_corr", "pred_corrupted", "yhat_corr"]
Y_KEYS = ["y_clean", "y_true", "label", "y"]

# Where your stage02 pt files live (adjust if needed)
PT_ROOT = PATHS.runs  # we’ll search below this



def get_nested(obj, path, default=None):
    """Safely get nested keys: get_nested(obj, ['corr_ref','pred'])."""
    cur = obj
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def pick_key(d: dict, candidates: list[str], required: bool = True) -> str | None:
    for k in candidates:
        if k in d:
            return k
    if required:
        raise KeyError(f"None of these keys found: {candidates}. Available keys: {list(d.keys())[:50]}")
    return None


def parse_corr_sev_from_path(p: Path) -> tuple[str | None, int | None]:
    s = p.as_posix()
    name = p.stem

    # 0) Your pattern: *__<corr>__sev<k>* (works for dirs or filenames)
    m = re.search(r"__(artifacts|drift)__([A-Za-z0-9_]+)__sev(\d+)", s)
    if m:
        return m.group(2), int(m.group(3))

    # 1) filename: corr<name>...sev<k>
    m = re.search(r"corr([A-Za-z0-9_]+).*sev(\d+)", name)
    if m:
        return m.group(1), int(m.group(2))

    # 2) filename: <name>...sev<k>
    m = re.search(r"([A-Za-z0-9_]+).*sev(?:erity)?(\d+)", name)
    if m:
        return m.group(1), int(m.group(2))

    # 3) path: corruption=<name>, severity=<k>
    m = re.search(r"corruption[=/]([A-Za-z0-9_]+)", s)
    n = re.search(r"severity[=/](\d+)", s)
    if m and n:
        return m.group(1), int(n.group(1))

    return None, None


def to_np(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def load_reference(ref_pt: Path):
    ref = torch.load(ref_pt, map_location="cpu")

    # Try common label locations (top-level + nested)
    y = (
        get_nested(ref, ["y"]) or
        get_nested(ref, ["y_true"]) or
        get_nested(ref, ["y_clean"]) or
        get_nested(ref, ["clean_reference", "y"]) or
        get_nested(ref, ["clean_reference", "y_true"]) or
        get_nested(ref, ["clean_reference", "y_clean"])
    )
    if y is None:
        # THIS PRINT IS THE NEXT THING YOU NEED
        print("Reference top-level keys:", list(ref.keys()) if isinstance(ref, dict) else type(ref))
        if isinstance(ref, dict) and "clean_ref" in ref and isinstance(ref["clean_ref"], dict):
            print("Reference clean_ref keys:", list(ref["clean_ref"].keys()))
        raise KeyError(f"{ref_pt}: couldn't find labels. See printed keys above.")

    y = to_np(y).astype(int).reshape(-1)

    pair_idx = (
        get_nested(ref, ["pair_idx"])
    )
    if pair_idx is not None:
        pair_idx = to_np(pair_idx).astype(int).reshape(-1)
    else:
        pair_idx = np.arange(len(y), dtype=int)
    
    proba_clean = ref["clean_reference"]["proba_clean"]   # saved by you
    if torch.is_tensor(proba_clean):
        proba_clean = proba_clean.detach().cpu().numpy()
    msp_clean = proba_clean.max(axis=1).astype(float).reshape(-1)

    return pair_idx, y, msp_clean


def load_artifact_pred(artifact_pt: Path):
    art = torch.load(artifact_pt, map_location="cpu")

    corr_ref = get_nested(art, ["corrupt_reference"])
    if corr_ref is None:
        raise KeyError(f"{artifact_pt}: missing 'corrupt_reference'")

    pred = get_nested(corr_ref, ["pred_corr"])
    if pred is None:
        raise KeyError(f"{artifact_pt}: missing corrupt_reference['pred_corr']")
    pred = to_np(pred).astype(int).reshape(-1)

    proba = get_nested(corr_ref, ["proba_corr"])
    msp = None
    if proba is not None:
        proba = to_np(proba)
        msp = proba.max(axis=1).astype(float).reshape(-1)

    return pred, msp


def load_stage02_vectors(drift_pt: Path):
    """Stage02 drift vectors: you already saw keys like exp__cosine_sim, exp__iou_topk, exp__spearman_rho, dH, invariant, both_correct."""
    obj = torch.load(drift_pt, map_location="cpu")
    vectors = obj["vectors"] if isinstance(obj, dict) and "vectors" in obj else obj

    dH = to_np(vectors["dH"]).astype(float).reshape(-1)
    inv = to_np(vectors["invariant"]).astype(bool).reshape(-1)
    bc  = to_np(vectors["both_correct"]).astype(bool).reshape(-1)

    cos = to_np(vectors["exp__cosine_sim"]).astype(float).reshape(-1)
    iou = to_np(vectors["exp__iou_topk"]).astype(float).reshape(-1)
    rho = to_np(vectors["exp__spearman_rho"]).astype(float).reshape(-1)

    return dH, inv, bc, cos, iou, rho



def build_pairs_table_from_pts(
    ref_pt: Path,
    artifacts_root: Path,
    drift_root: Path,
    out_csv: Path,
):
    pair_idx, y_true, msp_clean = load_reference(ref_pt)

    # index artifacts by (corr, sev)
    art_files = sorted(artifacts_root.rglob("*.pt"))
    art_map = {}
    for p in art_files:
        corr, sev = parse_corr_sev_from_path(p)
        if corr is None or sev is None:
            continue
        art_map[(corr, sev)] = p

    # index drift by (corr, sev)
    drift_files = sorted(drift_root.rglob("*.pt"))
    drift_map = {}
    for p in drift_files:
        corr, sev = parse_corr_sev_from_path(p)
        if corr is None or sev is None:
            continue
        drift_map[(corr, sev)] = p


    keys = sorted(set(art_map.keys()) & set(drift_map.keys()))
    if not keys:
        raise RuntimeError("No matching (corruption,severity) pairs found between artifacts and drift .pt files. Check parse_corr_sev().")

    rows = []
    for (corr, sev) in keys:
        pred_corr, msp_corr = load_artifact_pred(art_map[(corr, sev)])
        dH, inv, bc, cos, iou, rho = load_stage02_vectors(drift_map[(corr, sev)])

        N = len(y_true)

        # sanity: everything must align
        for name, vec in [
            ("pred_corr", pred_corr), ("dH", dH), ("inv", inv), ("bc", bc), ("cos", cos), ("iou", iou), ("rho", rho)
        ]:
            if len(vec) != N:
                raise ValueError(f"[{corr} sev{sev}] length mismatch: {name} has {len(vec)} but y_true has {N}")

        correct_corr = (pred_corr == y_true)

        df = pd.DataFrame({
            "corruption": [corr] * N,
            "severity": [int(sev)] * N,
            "pair_idx": pair_idx,

            "y_true": y_true,
            "pred_corr": pred_corr,
            "correct_corr": correct_corr,
            "wrong_corr": ~correct_corr,

            "msp_corr": msp_corr if msp_corr is not None else np.nan,
            "msp_clean": msp_clean if msp_clean is not None else np.nan,

            "dH": dH,
            "abs_dH": np.abs(dH),

            "invariant": inv,
            "both_correct": bc,

            "cos": cos,
            "iou": iou,
            "rho": rho,

            "drift_1m_cos": 1.0 - cos,
            "drift_1m_iou": 1.0 - iou,
            "drift_1m_rho": 1.0 - rho,
        })
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print("Wrote:", out_csv, "rows:", len(out))
    return out

# ---------- Spearman (no scipy) ----------
def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def load_pairs(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["severity"] = pd.to_numeric(df["severity"], errors="raise").astype(int)
    df["corruption"] = df["corruption"].astype(str)
    # ensure bool
    for b in ["invariant", "both_correct"]:
        if b in df.columns:
            df[b] = df[b].astype(bool)
    return df


def make_xy(df: pd.DataFrame, y_mode: str = "cos") -> tuple[np.ndarray, np.ndarray, str]:
    x = df["abs_dH"].to_numpy()
    if y_mode == "cos":
        y = 1.0 - df["cos"].to_numpy()
        ylab = "1 − cosine similarity (higher = more drift)"
    elif y_mode == "iou":
        y = 1.0 - df["iou"].to_numpy()
        ylab = "1 − IoU@topk (higher = more drift)"
    else:
        raise ValueError("y_mode must be 'cos' or 'iou'")
    return x, y, ylab


def scatter_plot(
    df: pd.DataFrame,
    title: str,
    y_mode: str = "cos",
    max_points: int = 8000,
    save_path: Path | None = None,
) -> None:
    d = df.dropna(subset=["abs_dH", "cos", "iou"]).copy()

    # downsample for readability
    if len(d) > max_points:
        d = d.sample(max_points, random_state=0)

    x, y, ylab = make_xy(d, y_mode=y_mode)
    rho = spearman_r(x, y)

    plt.figure(figsize=(7.8, 5.6))
    plt.scatter(x, y, s=10, alpha=0.35)
    plt.xlabel("|ΔEntropy| per pair")
    plt.ylabel(ylab)
    plt.title(f"{title}\nSpearman ρ = {rho:.3f}, n={len(d)}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.show()


def plotC_pooled_and_grouped(
    df: pd.DataFrame,
    slice_name: str = "invariant",      # "all" | "invariant" | "both_correct"
    y_mode: str = "cos",                # "cos" | "iou"
    grouped: bool = True,               # also make per (corr×sev) plots
    save_dir: Path | None = None,
) -> None:
    # choose slice
    if slice_name == "all":
        d = df.copy()
    else:
        if slice_name not in df.columns:
            raise ValueError(f"slice_name='{slice_name}' but column not found in pairs table.")
        d = df[df[slice_name]].copy()

    # pooled scatter
    scatter_plot(
        d,
        title=f"Plot C (pooled): |ΔEntropy| vs ΔE drift — slice={slice_name}, y={y_mode}",
        y_mode=y_mode,
        save_path=None if save_dir is None else (save_dir / f"plotC_pooled__{slice_name}__{y_mode}.png"),
    )

    if not grouped:
        return

    # per corruption×severity
    for (corr, sev), g in d.groupby(["corruption", "severity"]):
        scatter_plot(
            g,
            title=f"Plot C: {corr}, severity={sev} — slice={slice_name}, y={y_mode}",
            y_mode=y_mode,
            max_points=4000,
            save_path=None if save_dir is None else (save_dir / f"plotC__{corr}__sev{sev}__{slice_name}__{y_mode}.png"),
        )


def spearman_table(
    df: pd.DataFrame,
    slice_name: str = "invariant",
    y_mode: str = "cos",
) -> pd.DataFrame:
    # choose slice
    if slice_name == "all":
        d = df.copy()
    else:
        d = df[df[slice_name]].copy()

    rows = []
    # per corr×sev
    for (corr, sev), g in d.groupby(["corruption", "severity"]):
        x, y, _ = make_xy(g, y_mode=y_mode)
        rho = spearman_r(x, y)
        rows.append({"corruption": corr, "severity": int(sev), "rho": rho, "n": int(len(g))})

    out = pd.DataFrame(rows).sort_values(["corruption", "severity"])

    # pooled per corruption
    pooled = []
    for corr, g in d.groupby(["corruption"]):
        x, y, _ = make_xy(g, y_mode=y_mode)
        rho = spearman_r(x, y)
        pooled.append({"corruption": corr, "severity": "pooled", "rho": rho, "n": int(len(g))})

    return pd.concat([out, pd.DataFrame(pooled)], ignore_index=True)

def filter_slice(df: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    if slice_name == "all":
        return df.copy()
    if slice_name not in df.columns:
        raise ValueError(f"Slice '{slice_name}' not in df.columns.")
    return df[df[slice_name]].copy()


def violin_by_severity(
    df_pairs: pd.DataFrame,
    y_col: str = "rho",
    slice_name: str = "invariant",
    corruption: str | None = None,
    severities=(0, 1, 2, 3, 5),
    title: str | None = None,
    save_path: str | Path | None = None,
):
    d = df_pairs.copy()
    d["severity"] = pd.to_numeric(d["severity"], errors="raise").astype(int)
    d["corruption"] = d["corruption"].astype(str)

    d = filter_slice(d, slice_name)
    if corruption is not None:
        d = d[d["corruption"] == corruption].copy()

    d = d[d["severity"].isin(severities)].dropna(subset=[y_col]).copy()

    # build per-severity arrays, but KEEP ONLY non-empty ones
    sev_data = []
    sev_used = []
    ns = []
    for s in severities:
        arr = d.loc[d["severity"] == s, y_col].to_numpy()
        if arr.size == 0:
            continue
        sev_data.append(arr)
        sev_used.append(s)
        ns.append(arr.size)

    if len(sev_data) == 0:
        print(f"[violin] no data for corruption={corruption} slice={slice_name} y_col={y_col}")
        return

    fig, ax = plt.subplots(figsize=(9, 4.8))

    pos = np.arange(len(sev_used))
    ax.violinplot(
        sev_data,
        positions=pos,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    ax.boxplot(sev_data, positions=pos, widths=0.15, showfliers=False)

    ax.set_xticks(pos)
    ax.set_xticklabels([str(s) for s in sev_used])
    ax.set_xlabel("Severity")
    ax.set_ylabel(y_col)

    if title is None:
        scope = corruption if corruption else "pooled"
        title = f"Stability distribution: {y_col} vs severity — slice={slice_name}, {scope}"
    ax.set_title(title)

    # n labels
    y0, y1 = ax.get_ylim()
    for i, n in enumerate(ns):
        ax.text(i, y0, f"n={n}", ha="center", va="bottom", fontsize=9)

    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()



def main():
    # Find pt files under runs that look like drift vector dumps
    # Adjust glob pattern to your actual saved filenames.
    pt_files = sorted(PT_ROOT.rglob("02__drift/*.pt"))
    if not pt_files:
        pt_files = sorted(PT_ROOT.rglob("02__drift/**/*.pt"))

    exp_dir = PATHS.runs / "experiment__n250__IG__seed51"
    ref_pt = exp_dir / "00__reference" / "00__clean_ref.pt"          # adjust filename
    art_root = exp_dir / "01__artifacts"
    drift_root = exp_dir / "02__drift"
    
    out_dir = PATHS.results / "spearman_entropy"

    print(f"Found {len(pt_files)} .pt files under {PT_ROOT}")
    out_csv = out_dir / "pairs_table_from_pt.csv"
    df_pairs = build_pairs_table_from_pts(ref_pt, art_root, drift_root, out_csv)
    print("Wrote:", out_csv)
    print("Rows:", len(df_pairs), "Cols:", len(df_pairs.columns))
    print(df_pairs.head())

    
    pairs_csv = out_dir / "pairs_table_from_pt.csv"  # from your .pt extraction
    df_pairs = load_pairs(pairs_csv)

    # --- choose what you want ---
    # slice_name: "all" | "invariant" | "both_correct"
    # y_mode: "cos" (1-cos) or "iou" (1-iou)
    y_mode = "iou"

    plotC_pooled_and_grouped(
        df_pairs,
        slice_name="invariant",
        y_mode=y_mode,
        grouped=False,   # set True if you want one image per corr×sev (can be many)
        save_dir=out_dir,
    )

    tab = spearman_table(df_pairs, slice_name="invariant", y_mode=y_mode)
    tab.to_csv(out_dir / f"spearman_table__invariant__{y_mode}.csv", index=False)
    print(tab.head(30))

    for corr in ["brightness", "fog", "gaussian_noise"]:
        violin_by_severity(
            df_pairs,
            y_col="rho",
            slice_name="invariant",
            corruption=corr,
            save_path=out_dir / f"violin_rho__inv__{corr}.png",
        )


if __name__ == "__main__":
    main()
