"""
S11 - computational cost of the forecast models.

Motivation. A forecast system is not only a skill score. It is also a bill: the
hardware it needs, the time it occupies that hardware, and the energy that
implies. The event-skill result in this paper is that a model trained on
reanalysis meteorology beats an operational composition reanalysis at flagging
observer-reported dust days. That claim is more useful, not less, if the cost of
the cheaper system is stated rather than assumed.

What is measured, on the machine that produced every other result in the paper:

    fit_wall_s      wall-clock seconds to train, median of REPEATS runs
    fit_cpu_s       process CPU seconds over the same fit, summed across threads
    predict_ms      wall-clock milliseconds to produce the full test forecast
    predict_us_day  the same divided by the number of forecast days
    peak_rss_mb     peak resident set size during the fit, above the pre-fit
                    baseline, sampled at 20 ms
    model_mb        pickled size of the fitted estimator

What is NOT measured, and must not be claimed:

    Package power is not readable on this platform without a kernel driver, so
    energy is an estimate, not a measurement. It is computed as CPU-seconds
    times an assumed average package power and reported as a band between the
    processor's sustained power limit and its turbo limit. The band is wide on
    purpose. Anything narrower would be false precision.

    The cost of the reanalysis baseline is not measured here. It is produced on
    an operational supercomputer that this work has no access to. Any comparison
    is a citation, not a measurement, and is made in the discussion with that
    caveat attached.

Writes results/s11_compute_cost.csv and results/s11_report.txt.
"""

import io
import os
import pickle
import platform
import threading
import time

import numpy as np
import pandas as pd
import psutil

from s4_models import DROP, SEED, TEST, TRAIN, VAL, build_models

ROOT = os.environ.get(
    "ICIEAHS_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
RESULTS = os.path.join(ROOT, "results")

REPEATS = 5

# Intel Core i7-1255U. PL1 is the power the package is held to under sustained
# load; PL2 is the short-burst ceiling. Published values, not measured here.
POWER_SUSTAINED_W = 15.0
POWER_TURBO_W = 55.0

# Green Algorithms constants (Lannelongue, Grealey & Inouye, 2021). Memory draws
# power whether or not it is being written to, so it enters the energy budget as
# a function of the memory reserved and the time it is held.
MEMORY_W_PER_GB = 0.3725
# A laptop has no data-centre overhead: no chilled water, no redundant supply.
PUE = 1.0
# Ember, Middle East average for 2024. Iraq has essentially no wind or solar in
# its generation mix, so a regional average is the right order for a machine
# running in Duhok.
CARBON_G_PER_KWH = 636.0


LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


class PeakRSS:
    """Sample resident set size in a background thread.

    Peak memory is the number that decides whether a model runs on the hardware
    a forecast office actually has. It cannot be read after the fact, because the
    allocator returns the memory, so it has to be sampled while the fit runs.
    Sampling at 20 ms is fast enough to catch an allocation that lasts a tenth of
    a second and cheap enough not to perturb the timing it is measuring.
    """

    def __init__(self, interval=0.02):
        self.interval = interval
        self.proc = psutil.Process()
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self.baseline = self.proc.memory_info().rss
        self.peak = self.baseline
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self.proc.memory_info().rss)
            except Exception:
                return
            self._stop.wait(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def delta_mb(self):
        return (self.peak - self.baseline) / 1024 ** 2

    @property
    def peak_mb(self):
        """Absolute peak, not the increment above the pre-fit baseline.

        The increment is close to zero for the cheap models, because the fit
        runs inside heap the allocator has already claimed and never returned.
        A reader who sees 0.0 MB will conclude the measurement is broken, and
        will be right to. The absolute peak answers the question a forecast
        office actually asks: how much memory does this need?
        """
        return self.peak / 1024 ** 2


def pickled_mb(obj):
    buf = io.BytesIO()
    pickle.dump(obj, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.tell() / 1024 ** 2


def energy_wh(cpu_seconds, watts, wall_seconds=0.0, mem_gb=0.0):
    """Energy in watt-hours, following the Green Algorithms accounting.

    CPU-seconds are thread-seconds. Package power covers the whole package
    regardless of how many threads are busy, so dividing by the logical core
    count converts thread-seconds back to package-seconds before applying the
    package power. Written this way the core term is exactly the Green
    Algorithms product of runtime, core count, per-core power and a usage
    factor, with the usage factor measured rather than assumed.

    The memory term is separate because DRAM draws power for as long as it is
    held, not only while it is being touched.
    """
    package_seconds = cpu_seconds / (os.cpu_count() or 1)
    core_wh = package_seconds * watts / 3600.0
    mem_wh = wall_seconds * mem_gb * MEMORY_W_PER_GB / 3600.0
    return (core_wh + mem_wh) * PUE


def carbon_g(wh):
    return wh / 1000.0 * CARBON_G_PER_KWH


def main():
    df = pd.read_csv(
        os.path.join(RESULTS, "s1_daily.csv"), parse_dates=["date"]
    ).set_index("date")

    say("=" * 70)
    say("S11 COMPUTATIONAL COST")
    say("=" * 70)
    say()
    say(f"platform        {platform.platform()}")
    say(f"processor       12th Gen Intel(R) Core(TM) i7-1255U")
    say(f"logical cores   {os.cpu_count()}")
    say(f"python          {platform.python_version()}")
    import sklearn
    say(f"scikit-learn    {sklearn.__version__}")
    say(f"repeats         {REPEATS} fits per model, median reported")
    say()

    features_all = [c for c in df.columns if c not in DROP]
    cams_cols = [c for c in features_all if c.startswith("cams_")]
    lag_cols = [f"aod550_lag{i}" for i in range(1, 6)]
    usable = df.dropna(subset=["aod550"] + lag_cols)

    feature_sets = {
        "meteo_only": [c for c in features_all if c not in cams_cols],
        "plus_reanalysis": features_all,
    }

    rows = []
    for set_name, cols in feature_sets.items():
        data = usable.dropna(subset=cols) if set_name == "plus_reanalysis" else usable
        sub = data[cols + ["aod550"]].copy()
        sub[cols] = sub[cols].ffill(limit=1)
        sub = sub.dropna()

        tr = sub[sub.index.year.isin(TRAIN)]
        te = sub[sub.index.year.isin(TEST)]
        x_tr, y_tr = tr[cols].to_numpy(), tr["aod550"].to_numpy()
        x_te = te[cols].to_numpy()

        say(f"--- {set_name}: {len(cols)} predictors, "
            f"{len(tr)} training days, {len(te)} forecast days ---")

        for model_name in build_models():
            walls, cpus, peaks = [], [], []
            fitted = None
            for _ in range(REPEATS):
                model = build_models()[model_name]
                with PeakRSS() as rss:
                    t0, c0 = time.perf_counter(), time.process_time()
                    model.fit(x_tr, y_tr)
                    walls.append(time.perf_counter() - t0)
                    cpus.append(time.process_time() - c0)
                peaks.append(rss.peak_mb)
                fitted = model

            # Inference is timed separately and repeated, because a single
            # prediction on a thousand rows is short enough that process
            # scheduling noise is the same order as the quantity.
            preds = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                fitted.predict(x_te)
                preds.append(time.perf_counter() - t0)

            wall = float(np.median(walls))
            cpu = float(np.median(cpus))
            pred_s = float(np.median(preds))
            rows.append({
                "feature_set": set_name,
                "model": model_name,
                "n_predictors": len(cols),
                "n_train": len(tr),
                "n_forecast_days": len(te),
                "fit_wall_s": round(wall, 3),
                "fit_wall_s_min": round(float(np.min(walls)), 3),
                "fit_wall_s_max": round(float(np.max(walls)), 3),
                "fit_cpu_s": round(cpu, 3),
                "predict_ms": round(pred_s * 1e3, 2),
                "predict_us_day": round(pred_s * 1e6 / len(te), 1),
                "peak_rss_mb": round(float(np.median(peaks)), 1),
                "model_mb": round(pickled_mb(fitted), 2),
                "energy_wh_lo": round(energy_wh(
                    cpu, POWER_SUSTAINED_W, wall,
                    float(np.median(peaks)) / 1024), 5),
                "energy_wh_hi": round(energy_wh(
                    cpu, POWER_TURBO_W, wall,
                    float(np.median(peaks)) / 1024), 5),
            })
            r = rows[-1]
            say(f"  {model_name:19} fit {r['fit_wall_s']:7.3f} s wall "
                f"[{r['fit_wall_s_min']:.2f}-{r['fit_wall_s_max']:.2f}]  "
                f"{r['fit_cpu_s']:8.3f} s CPU   "
                f"predict {r['predict_ms']:7.2f} ms   "
                f"peak {r['peak_rss_mb']:6.1f} MB   "
                f"model {r['model_mb']:6.2f} MB")
        say()

    cost = pd.DataFrame(rows)
    cost.to_csv(os.path.join(RESULTS, "s11_compute_cost.csv"), index=False)

    head = cost[(cost.feature_set == "meteo_only")]
    total_cpu = float(head.fit_cpu_s.sum())
    e_lo = float(head.energy_wh_lo.sum())
    e_hi = float(head.energy_wh_hi.sum())
    say("=" * 70)
    say("HEADLINE CONFIGURATION, meteorology-only feature set")
    say("=" * 70)
    say(f"  training the three models once costs {total_cpu:.1f} CPU-seconds "
        f"in total")
    say(f"  estimated energy, at {POWER_SUSTAINED_W:.0f} to "
        f"{POWER_TURBO_W:.0f} W package power: "
        f"{e_lo*1000:.1f} to {e_hi*1000:.1f} mWh")
    say(f"  at {CARBON_G_PER_KWH:.0f} gCO2e/kWh: "
        f"{carbon_g(e_lo):.2f} to {carbon_g(e_hi):.2f} gCO2e")
    rf = head[head.model == "random_forest"].iloc[0]
    say(f"  the reported model, random forest, produces a full "
        f"{int(rf.n_forecast_days)}-day forecast series in "
        f"{rf.predict_ms:.1f} ms, or {rf.predict_us_day:.1f} microseconds per "
        f"forecast day")

    # Published reference point. Not a controlled comparison and must never be
    # presented as one: a global 9 km numerical forecast and a single-domain
    # statistical forecast solve different problems. The figure is quoted to
    # establish the order of magnitude that a supercomputer-based operational
    # system occupies.
    ifs_g_per_forecast_day = 1660.0
    say()
    say(f"  reference point, Lannelongue et al. (2021): one IFS forecast day is")
    say(f"  estimated at {ifs_g_per_forecast_day:.0f} gCO2e on 4,608 cores.")
    say(f"  ratio to training all three models here: "
        f"{ifs_g_per_forecast_day/carbon_g(e_hi):,.0f} to "
        f"{ifs_g_per_forecast_day/carbon_g(e_lo):,.0f} times")
    say()
    say("  Energy figures are estimates from measured CPU time and published")
    say("  package power limits, not wall-socket measurements. The cost of the")
    say("  operational reanalysis baseline is not measured here, and the IFS")
    say("  figure above is a published estimate for a different problem.")
    say()

    with open(os.path.join(RESULTS, "s11_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    say(f"wrote {os.path.join(RESULTS, 's11_compute_cost.csv')}")


if __name__ == "__main__":
    main()
