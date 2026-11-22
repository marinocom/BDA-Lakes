"""Fit the extended models and the train/test split, then export the web app data.

Run from the project folder with the course environment active:

    python run_extended.py            # everything, skipping fits that already exist
    python run_extended.py app split  # only some stages

Stages, in the order they run:
    m4      M4 (M3 + altitude) on the main data
    app     export M4 posterior draws for the lake predictor (webui/posterior_draws.js)
    split   M0-M4 fitted on the training lakes and years, then scored on the three test sets
    refit   M1 and M2 refitted with 2000 warm-up + 2000 draws per chain (they mixed slowly)
    sens    M4 with all prior scales x0.5 and x2
    lake    M4 and M5 (M4 + lake area + depth) on the 53 sites with reliable lake data

Progress goes to results/run_extended.log. Each fit is saved to results/ as soon as it
finishes, so the script can be stopped and restarted.
"""
import glob
import json
import os
import shutil
import sys
import time
import warnings


def _find_cxx():
    """PyTensor needs a C++ compiler; outside an activated conda shell it may not find one.
    Look for the conda environment's own compiler first, then the system ones."""
    cands = [os.environ.get("CXX")]
    cands += sorted(glob.glob(os.path.join(sys.prefix, "bin", "*clang++")))
    cands += sorted(glob.glob(os.path.join(sys.prefix, "bin", "*g++")))
    cands += [shutil.which("clang++"), shutil.which("g++")]
    for c in cands:
        if c and os.path.exists(c if os.path.isabs(c) else shutil.which(c) or ""):
            return c
    return None


if "cxx=" not in os.environ.get("PYTENSOR_FLAGS", ""):
    _cxx = _find_cxx()
    if _cxx:
        os.environ["PYTENSOR_FLAGS"] = ",".join(
            f for f in [os.environ.get("PYTENSOR_FLAGS", ""), f"cxx={_cxx}"] if f)

import numpy as np
import pandas as pd

import icebreak_common as ic

warnings.filterwarnings("ignore", message="libc not found")

LOG = ic.RES_DIR / "run_extended.log"
SPLIT_DIR = ic.RES_DIR / "split"
SPLIT_DIR.mkdir(exist_ok=True)
WEB_DIR = ic.ROOT / "webui"
WEB_DIR.mkdir(exist_ok=True)
LONG = dict(draws=2000, tune=2000)          # for M1 and M2, which mix slowly


def log(msg):
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fit(name, data, path, prior_scale=1.0, force=False, **kw):
    """Fit (or load) a model and log how it went."""
    if path.exists() and not force:
        log(f"{path.name}: already there, loading")
        return ic.build_model(name, data, prior_scale), ic.fit_or_load(None, path)
    log(f"{path.name}: fitting {name} on {len(data):,} rows ...")
    t0 = time.time()
    m = ic.build_model(name, data, prior_scale)
    idata = ic.fit_or_load(m, path, force=True, **kw)
    d = ic.diagnostics_table(idata)
    log(f"{path.name}: done in {(time.time() - t0) / 60:.1f} min | max R-hat {d['max R-hat']:.3f}"
        f" | min bulk-ESS {d['min bulk-ESS']:.0f} | divergences {d['divergences']}")
    return m, idata


def n_draws(idata):
    return idata.posterior.sizes["draw"]


# ---------------------------------------------------------------- stages
def stage_m4(d):
    fit("M4_altitude", d, ic.RES_DIR / "M4_altitude.nc")


def stage_app(d):
    """Posterior draws of M4 for the browser app (no server needed)."""
    import arviz as az

    path = ic.RES_DIR / "M4_altitude.nc"
    if not path.exists():
        log("app: M4 not fitted yet, skipping")
        return
    post = az.from_netcdf(path).posterior
    S = post.sizes["chain"] * post.sizes["draw"]
    keep = np.linspace(0, S - 1, 600).astype(int)
    scal = ["Intercept", "lat_c", "lon_c", "alt_c", "decade_c", "post1980", "lat_c:decade_c",
            "1|site_sigma", "decade_c|site_sigma", "1|season_f_sigma", "1|season_region_sigma",
            "sigma", "nu"]
    draws = {v: np.round(ic.draws(post, v)[keep], 4).tolist() for v in scal}
    a = ic.draws(post, "1|site")[keep]
    b = ic.draws(post, "decade_c|site")[keep]
    levels = [str(x) for x in post["1|site"][post["1|site"].dims[-1]].values]
    info = d.groupby("site").agg(name=("Nimi", "first"), lat=("lat", "first"), lon=("lon", "first"),
                                 alt=("lake_altitude_m", "first"), area=("lake_area_km2", "first"),
                                 n=("season", "size"), first=("season", "min"),
                                 last=("season", "max"), mean_doy=(ic.RESPONSE, "mean"))
    sites = []
    for j, s in enumerate(levels):
        r = info.loc[s]
        recent = d[(d.site == s) & (d.season >= 2010)][ic.RESPONSE]
        sites.append({"id": s, "name": r["name"], "lat": round(r.lat, 4), "lon": round(r.lon, 4),
                      "alt": round(float(r.alt), 1),
                      "area": None if pd.isna(r.area) else round(float(r.area), 1),
                      "n": int(r.n), "first": int(r["first"]), "last": int(r["last"]),
                      "mean_doy": round(float(r.mean_doy), 1),
                      "recent_doy": None if recent.empty else round(float(recent.mean()), 1),
                      "a": np.round(a[:, j], 2).tolist(), "b": np.round(b[:, j], 3).tolist()})
    payload = {"model": "M4_altitude", "lat_center": ic.LAT_CENTER, "lon_center": ic.LON_CENTER,
               "alt_center": ic.ALT_CENTER, "alt_scale": ic.ALT_SCALE,
               "break_year": ic.BREAK_YEAR, "region_cuts": ic.REGION_CUTS[1:3],
               "n_draws": len(keep), "draws": draws, "sites": sites,
               "data_range": {"lat": [float(d.lat.min()), float(d.lat.max())],
                              "lon": [float(d.lon.min()), float(d.lon.max())],
                              "alt": [float(d.lake_altitude_m.min()), float(d.lake_altitude_m.max())],
                              "season": [int(d.season.min()), int(d.season.max())]}}
    out = WEB_DIR / "posterior_draws.js"
    out.write_text("window.POSTERIOR = " + json.dumps(payload, ensure_ascii=False) + ";\n",
                   encoding="utf-8")
    log(f"app: wrote {out.relative_to(ic.ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


def predictive(model, idata, newdata, seed=1, n=1000):
    """Thinned posterior predictive draws (n x N) and pointwise log density for new data."""
    from scipy import stats
    from scipy.special import logsumexp

    import arviz as az

    S = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    post = idata.posterior.sel(draw=slice(None, None, max(1, S // n)))
    # renumber the kept draws 0..k-1: Bambi numbers the new-group draws that way, and
    # xarray would otherwise misalign them with the thinned draw labels
    post = post.assign_coords(draw=np.arange(post.sizes["draw"]))
    thin = az.InferenceData(posterior=post)
    pred = ic.predict(model, thin, "response_params", data=newdata,
                      sample_new_groups=True, inplace=False)
    mu = ic.mu_draws(pred)
    sig = ic.draws(thin.posterior, "sigma")[:, None]
    nu = ic.draws(thin.posterior, "nu")[:, None]
    y = newdata[ic.RESPONSE].to_numpy()[None, :]
    ll = stats.t.logpdf(y, df=nu, loc=mu, scale=sig)
    lpd = logsumexp(ll, axis=0) - np.log(ll.shape[0])
    rng = np.random.default_rng(seed)
    yrep = mu + sig * rng.standard_t(np.broadcast_to(nu, mu.shape))
    return yrep, lpd


def stage_split(d):
    sp = ic.make_split(d)
    log("split: " + ", ".join(f"{k} {len(v):,} rows / {v.site.nunique()} lakes"
                              for k, v in sp.items()))
    rows, preds = [], []
    names = ["M0_pooled", "M1_hier_linear", "M2_hier_broken", "M3_hier_broken_region", "M4_altitude"]
    for name in names:
        kw = LONG if name in ("M1_hier_linear", "M2_hier_broken") else {}
        m, idata = fit(name, sp["train"], SPLIT_DIR / f"{name}.nc", **kw)
        m.build()
        for tset in ["A_future", "B_lakes", "C_both"]:
            nd = sp[tset]
            yrep, lpd = predictive(m, idata, nd)
            y = nd[ic.RESPONSE].to_numpy()
            med = np.median(yrep, 0)
            q = np.percentile(yrep, [5, 25, 75, 95], axis=0)
            rows.append({"model": name, "test_set": tset, "n": len(y),
                         "elpd": lpd.sum(), "elpd_se": np.sqrt(len(y)) * lpd.std(),
                         "elpd_per_obs": lpd.mean(),
                         "MAE_days": np.mean(np.abs(med - y)),
                         "RMSE_days": np.sqrt(np.mean((yrep.mean(0) - y) ** 2)),
                         "cover50": np.mean((y >= q[1]) & (y <= q[2])),
                         "cover90": np.mean((y >= q[0]) & (y <= q[3])),
                         "width90_days": np.mean(q[3] - q[0])})
            preds.append(pd.DataFrame({"model": name, "test_set": tset, "site": nd.site,
                                       "name": nd.Nimi, "season": nd.season, "lat": nd.lat,
                                       "y": y, "median": med, "q5": q[0], "q95": q[3],
                                       "lpd": lpd}))
            log(f"split: {name} on {tset}: MAE {rows[-1]['MAE_days']:.2f} d, "
                f"90% cover {rows[-1]['cover90']:.2f}, elpd {rows[-1]['elpd']:.1f}")
    # simple non-Bayesian baselines for context (point predictions only)
    tr = sp["train"]
    site_mean = tr.groupby("site")[ic.RESPONSE].mean()
    lat_fit = np.polyfit(tr.groupby("site").lat.first(), site_mean, 1)
    for tset in ["A_future", "B_lakes", "C_both"]:
        nd = sp[tset]
        y = nd[ic.RESPONSE].to_numpy()
        if tset == "A_future":
            guess = nd.site.map(site_mean).to_numpy()
            lab = "baseline: lake's training mean"
        else:
            guess = np.polyval(lat_fit, nd.lat.to_numpy())
            lab = "baseline: latitude line"
        rows.append({"model": lab, "test_set": tset, "n": len(y),
                     "MAE_days": np.mean(np.abs(guess - y)),
                     "RMSE_days": np.sqrt(np.mean((guess - y) ** 2))})
    pd.DataFrame(rows).to_csv(ic.RES_DIR / "split_metrics.csv", index=False)
    pd.concat(preds).to_csv(ic.RES_DIR / "split_predictions.csv", index=False)
    log("split: wrote results/split_metrics.csv and results/split_predictions.csv")


def stage_refit(d):
    import arviz as az

    for name in ["M1_hier_linear", "M2_hier_broken"]:
        path = ic.RES_DIR / f"{name}.nc"
        if path.exists() and n_draws(az.from_netcdf(path)) >= LONG["draws"]:
            log(f"{path.name}: already refitted with {LONG['draws']} draws")
            continue
        fit(name, d, path, force=True, **LONG)


def stage_sens(d):
    for sc in [0.5, 2.0]:
        fit("M4_altitude", d, ic.RES_DIR / f"M4_altitude_prior{sc}.nc", prior_scale=sc)


def stage_lake(d):
    r = ic.reliable_lake_subset(d)
    log(f"lake: reliable subset {len(r):,} rows, {r.site.nunique()} sites")
    fit("M4_altitude", r, ic.RES_DIR / "M4_altitude_reliable.nc")
    fit("M5_lake_size", r, ic.RES_DIR / "M5_lake_size.nc")


def stage_nb05(d):
    """Execute notebook 05 in place so its tables and figures are saved in the file."""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    path = ic.ROOT / "05_altitude_split.ipynb"
    nb = nbformat.read(path, as_version=4)
    log("nb05: running 05_altitude_split.ipynb ...")
    ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(ic.ROOT)}})
    nbformat.write(nb, path)
    log("nb05: done, outputs saved in the notebook")


STAGES = {"m4": stage_m4, "app": stage_app, "split": stage_split, "refit": stage_refit,
          "sens": stage_sens, "lake": stage_lake, "nb05": stage_nb05}

if __name__ == "__main__":
    todo = sys.argv[1:] or [k for k in STAGES if k != "nb05"]
    d = ic.load_breakup(recent_only=True)
    log(f"=== run_extended.py: {', '.join(todo)} ===")
    import pytensor
    log(f"python {sys.executable} | C++ compiler: {pytensor.config.cxx or 'NONE (slow!)'}")
    if not pytensor.config.cxx:
        log("No C++ compiler found, so sampling would be 50x slower. Run this from a terminal "
            "where `conda activate collegeEnv` has been done, or set PYTENSOR_FLAGS=cxx=/path/to/clang++.")
        sys.exit(1)
    t0 = time.time()
    for s in todo:
        try:
            STAGES[s](d)
        except Exception as e:          # keep going with the other stages
            import traceback
            log(f"stage {s} FAILED: {type(e).__name__}: {e}")
            with open(LOG, "a") as f:
                traceback.print_exc(file=f)
    log(f"=== finished in {(time.time() - t0) / 60:.0f} min ===")
