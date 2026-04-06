import torch



def load_clean_ref(exp_dir, keys):
    ref_pt  = exp_dir.reference / "00__clean_ref.pt"
    ref = torch.load(ref_pt, map_location="cpu")

    cr = ref["clean_reference"]
    res = {}
    for k in keys:
        if k not in cr:
            raise KeyError(f"Missing key '{k}' in clean_reference. Available: {list(cr.keys())[:20]} ...")
        res[k] = cr[k]
    return res


def load_corr_ref(exp_dir, corr, sev, keys):
    corr_pt = exp_dir.artifacts / f"01__artifacts__{corr}__sev{sev}.pt"
    art = torch.load(corr_pt, map_location="cpu", weights_only=False)

    cc = art["corrupt_reference"]
    res = {}
    for k in keys:
        if k not in cc:
            raise KeyError(f"Missing key '{k}' in corrupt_reference. Available: {list(cc.keys())[:20]} ...")
        res[k] = cc[k]
    return res


def load_drift_artifacts(exp_dir, corr, sev, keys):
    drift_pt = exp_dir.drift / f"02__drift_{corr}__sev{sev}.pt"
    ref = torch.load(drift_pt, map_location="cpu")

    res = []
    for key in keys: 
        res.append(ref[f"{key}"])
    
    return res