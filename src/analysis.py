"""Exploratory data analysis of the liver patient records, organised by syllabus module.

Run from the project root:  python src/analysis.py
Writes figures to figures/, tables to results/tables/ and every number quoted in
the report to results/results.json.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.metrics import (accuracy_score, adjusted_rand_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (GroupShuffleSplit, RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from xgboost import XGBClassifier

import style as S

warnings.filterwarnings("ignore")
S.apply()

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
TAB = ROOT / "results" / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)
R = {}  # every number the report quotes

COLS = ["age", "gender", "tb", "db", "alkphos", "sgpt", "sgot", "tp", "alb", "ag", "result"]
NUM = ["age", "tb", "db", "alkphos", "sgpt", "sgot", "tp", "alb", "ag"]
SKEWED = ["tb", "db", "alkphos", "sgpt", "sgot"]
LABEL = {"age": "Age", "tb": "Total bilirubin", "db": "Direct bilirubin", "alkphos": "Alkaline phosphatase",
         "sgpt": "ALT (SGPT)", "sgot": "AST (SGOT)", "tp": "Total protein", "alb": "Albumin",
         "ag": "A/G ratio", "male": "Male"}
UNIT = {"age": "years", "tb": "mg/dL", "db": "mg/dL", "alkphos": "IU/L", "sgpt": "IU/L", "sgot": "IU/L",
        "tp": "g/dL", "alb": "g/dL", "ag": "ratio"}


def tsave(df, name, index=True):
    df.to_csv(TAB / f"{name}.csv", index=index)


def log_frame(df):
    out = df[NUM].copy()
    for c in SKEWED:
        out[c] = np.log1p(out[c])
    return out


t0 = time.time()
# ---------------------------------------------------------------- Module 1
raw = pd.read_csv(ROOT / "data" / "liver.csv", encoding="latin1")
R["raw_header"] = [c.strip() for c in raw.columns]
raw.columns = COLS
orig = pd.read_csv(ROOT / "data" / "ilpd_original_583.csv", header=None, names=COLS)

R["m1"] = {
    "n_rows": int(len(raw)), "n_cols": int(raw.shape[1]),
    "result_counts_raw": {int(k): int(v) for k, v in raw.result.value_counts().items()},
    "gender_counts_raw": {str(k): int(v) for k, v in raw.gender.value_counts(dropna=False).items()},
    "n_orig": int(len(orig)),
}

# ---------------------------------------------------------------- Module 2: transformation
n_exact_dup = int(raw.duplicated().sum())
df = raw.drop_duplicates().reset_index(drop=True)
df["result"] = (df.result == 1).astype(int)          # 1 = liver disease, 0 = no disease
orig["result"] = (orig.result == 1).astype(int)
in_orig = df.merge(orig.drop_duplicates(), on=COLS, how="left", indicator=True)["_merge"].eq("both").values

orig_u = orig.drop_duplicates().reset_index(drop=True)

R["m2"] = {"n_exact_dup": n_exact_dup, "n_after_dedup": int(len(df)),
           "n_unique_features": int(df.drop(columns="result").drop_duplicates().shape[0]),
           "result_counts_dedup": {int(k): int(v) for k, v in df.result.value_counts().items()},
           "n_orig_rows_found": int(in_orig.sum())}

# duplicates by class, before and after
fig, ax = plt.subplots(figsize=(6.2, 2.4))
labels = ["Raw file", "After de-duplication", "Original UCI release"]
dis = [R["m1"]["result_counts_raw"][1], R["m2"]["result_counts_dedup"][1], int(orig.result.sum())]
nod = [R["m1"]["result_counts_raw"][2], R["m2"]["result_counts_dedup"][0], int((orig.result == 0).sum())]
y = np.arange(3)[::-1]
ax.barh(y, dis, color=S.OXBLOOD, height=0.55, label="Liver disease")
ax.barh(y, nod, left=dis, color=S.TEAL, height=0.55, label="No disease")
for yi, a, b in zip(y, dis, nod):
    ax.text(a + b + 250, yi, f"{a + b:,}  ({a / (a + b):.0%} disease)", va="center", fontsize=8, color=S.INK)
ax.set_yticks(y, labels)
ax.set_xlim(0, 39000)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Records")
ax.legend(loc="lower right", ncol=2)
S.save(fig, FIG / "f01_records.png")

# provenance: where do the values of the extra (non-original) rows come from?
synth = df[~in_orig]
prov = {}
for c in NUM:
    s1 = set(orig.loc[orig.result == 1, c].dropna())
    s0 = set(orig.loc[orig.result == 0, c].dropna())
    v = synth[[c, "result"]].dropna()
    same = np.where(v.result == 1, v[c].isin(s1), v[c].isin(s0)).mean()
    other = np.where(v.result == 1, v[c].isin(s0), v[c].isin(s1)).mean()
    prov[c] = {"in_original": float(v[c].isin(orig[c]).mean()), "same_class": float(same), "other_class": float(other),
               "distinct_values": int(df[c].nunique())}
R["m2"]["provenance"] = prov
R["m2"]["n_synthetic"] = int(len(synth))
R["m2"]["orig_unique"] = int(len(orig_u))
R["m2"]["orig_gender_rate"] = {k: float(v) for k, v in orig.groupby("gender").result.mean().items()}
R["m2"]["aug_gender_rate"] = {k: float(v) for k, v in df.groupby("gender").result.mean().items()}
R["m2"]["orig_age_corr"] = float(orig[["age", "result"]].corr().iloc[0, 1])
R["m2"]["aug_age_corr"] = float(df[["age", "result"]].corr().iloc[0, 1])
R["m2"]["orig_age_p"] = float(stats.mannwhitneyu(orig.age[orig.result == 1], orig.age[orig.result == 0])[1])
R["m2"]["orig_gender_p"] = float(stats.chi2_contingency(pd.crosstab(orig.gender, orig.result))[1])

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.5, 1]})
cs_ = [c for c in NUM]
x = np.arange(len(cs_))
axes[0].bar(x - 0.2, [prov[c]["same_class"] * 100 for c in cs_], 0.4, color=S.OXBLOOD, label="found among same-class originals")
axes[0].bar(x + 0.2, [prov[c]["other_class"] * 100 for c in cs_], 0.4, color=S.RULE, label="found among other-class originals")
axes[0].set_xticks(x, ["Age", "TBil", "DBil", "ALP", "ALT", "AST", "TP", "Alb", "A/G"])
axes[0].set_ylabel("Share of synthetic values (%)")
axes[0].set_ylim(0, 118)
axes[0].legend(loc="upper center", ncol=2, fontsize=7)
axes[0].grid(axis="x", visible=False)
axes[0].set_title("Where the values of the added rows come from")
grp = ["Female", "Male"]
xo = np.arange(2)
axes[1].bar(xo - 0.2, [R["m2"]["orig_gender_rate"][g] * 100 for g in grp], 0.4, color=S.TEAL, label="original 583")
axes[1].bar(xo + 0.2, [R["m2"]["aug_gender_rate"][g] * 100 for g in grp], 0.4, color=S.SAND, label="30k file")
for i, g in enumerate(grp):
    axes[1].text(i - 0.2, R["m2"]["orig_gender_rate"][g] * 100 + 1.5, f"{R['m2']['orig_gender_rate'][g]:.0%}", ha="center", fontsize=7.5)
    axes[1].text(i + 0.2, R["m2"]["aug_gender_rate"][g] * 100 + 1.5, f"{R['m2']['aug_gender_rate'][g]:.0%}", ha="center", fontsize=7.5)
axes[1].set_xticks(xo, grp)
axes[1].set_ylim(50, 85)
axes[1].set_ylabel("Disease rate (%)")
axes[1].legend(loc="upper left", fontsize=7)
axes[1].grid(axis="x", visible=False)
axes[1].set_title("Disease rate by gender")
fig.tight_layout()
S.save(fig, FIG / "f29_provenance.png")

# A/G ratio consistency: A/G should equal ALB / (TP - ALB)
ag_calc = df.alb / (df.tp - df.alb)
ok = (df.ag - ag_calc).abs() <= 0.1
valid = df[["ag", "alb", "tp"]].notna().all(axis=1) & (df.tp > df.alb)
R["m2"]["ag_consistent_share"] = float(ok[valid].mean())
R["m2"]["ag_checked"] = int(valid.sum())
R["m2"]["tp_le_alb"] = int((df.tp <= df.alb).sum())
R["m2"]["db_gt_tb"] = int((df.db > df.tb).sum())

# missing data
miss = pd.DataFrame({"missing": df.isna().sum(), "share": df.isna().mean()})
miss = miss.drop(index="result")
rate_obs, rate_mis, pvals = [], [], []
for c in miss.index:
    m = df[c].isna()
    rate_mis.append(df.loc[m, "result"].mean())
    rate_obs.append(df.loc[~m, "result"].mean())
    tab = pd.crosstab(m, df.result)
    pvals.append(stats.chi2_contingency(tab)[1])
miss["disease_rate_if_missing"] = rate_mis
miss["disease_rate_if_observed"] = rate_obs
miss["chi2_p"] = pvals
tsave(miss.round(4), "missing")
R["m2"]["rows_any_missing"] = int(df.isna().any(axis=1).sum())
R["m2"]["cells_missing"] = int(df.isna().sum().sum())
R["m2"]["max_missing_per_row"] = int(df.isna().sum(axis=1).max())
R["m2"]["missing"] = miss.round(4).to_dict(orient="index")
pattern_counts = df[NUM + ["gender"]].isna().sum(axis=1).value_counts().sort_index()
R["m2"]["missing_per_row_counts"] = {int(k): int(v) for k, v in pattern_counts.items()}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.2]})
order = miss.sort_values("share").index
lab = [LABEL.get(c, "Gender") for c in order]
axes[0].barh(lab, miss.loc[order, "share"] * 100, color=S.SLATE, height=0.6)
axes[0].set_xlabel("Missing (%)")
axes[0].grid(axis="y", visible=False)
axes[0].set_title("Share of values missing")
yy = np.arange(len(order))
axes[1].hlines(yy, miss.loc[order, "disease_rate_if_observed"] * 100, miss.loc[order, "disease_rate_if_missing"] * 100,
               color=S.RULE, lw=2)
axes[1].plot(miss.loc[order, "disease_rate_if_observed"] * 100, yy, "o", color=S.TEAL, label="value observed")
axes[1].plot(miss.loc[order, "disease_rate_if_missing"] * 100, yy, "o", color=S.OXBLOOD, label="value missing")
axes[1].set_yticks(yy, [])
axes[1].set_xlabel("Disease rate in those rows (%)")
axes[1].set_title("Does missingness track the outcome?")
axes[1].legend(loc="upper right")
axes[1].annotate("only 2 rows", (100, 0), xytext=(93, 1.2), fontsize=7, color=S.MUTED,
                 arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.6))
axes[1].grid(axis="y", visible=False)
S.save(fig, FIG / "f02_missing.png")


def em_mvn(X, max_iter=200, tol=1e-6):
    """Maximum-likelihood mean and covariance of a multivariate normal with missing
    values (EM algorithm). Returns conditional-mean imputations, mu, Sigma, iterations."""
    X = np.asarray(X, float)
    n, p = X.shape
    M = np.isnan(X)
    mu = np.nanmean(X, 0)
    Xf = np.where(M, mu, X)
    Sig = np.cov(Xf, rowvar=False)
    keys = pd.Series([m.tobytes() for m in M])
    groups = keys.groupby(keys).groups
    ll_path = []
    for it in range(max_iter):
        Xh = Xf.copy()
        C = np.zeros((p, p))
        for _, rows in groups.items():
            rows = np.asarray(rows)
            m = M[rows[0]]
            if not m.any():
                continue
            o = ~m
            if not o.any():
                Xh[np.ix_(rows, m)] = mu
                C += len(rows) * Sig
                continue
            Soo = Sig[np.ix_(o, o)]
            Smo = Sig[np.ix_(m, o)]
            B = Smo @ np.linalg.inv(Soo)
            Xh[np.ix_(rows, m)] = mu[m] + (X[np.ix_(rows, o)] - mu[o]) @ B.T
            C[np.ix_(m, m)] += len(rows) * (Sig[np.ix_(m, m)] - B @ Smo.T)
        mu_new = Xh.mean(0)
        D = Xh - mu_new
        Sig_new = (D.T @ D + C) / n
        delta = np.abs(mu_new - mu).max() + np.abs(Sig_new - Sig).max()
        mu, Sig, Xf = mu_new, Sig_new, Xh
        ll_path.append(delta)
        if delta < tol:
            break
    return Xf, mu, Sig, it + 1


# imputation benchmark: hide 10 % of known values in complete rows and score each method
L = log_frame(df)
comp = L.dropna().values
mu_c, sd_c = comp.mean(0), comp.std(0)
Z = (comp - mu_c) / sd_c
mask = rng.random(Z.shape) < 0.10
mask[mask.all(axis=1), 0] = False
Zm = Z.copy()
Zm[mask] = np.nan
imps = {
    "Mean": SimpleImputer(strategy="mean").fit_transform(Zm),
    "Median": SimpleImputer(strategy="median").fit_transform(Zm),
    "KNN (k=5)": KNNImputer(n_neighbors=5).fit_transform(Zm),
    "Iterative (chained regression)": IterativeImputer(random_state=SEED, max_iter=20).fit_transform(Zm),
    "EM, multivariate normal (MLE)": em_mvn(Zm)[0],
}
imp_rmse = {k: float(np.sqrt(np.mean((v[mask] - Z[mask]) ** 2))) for k, v in imps.items()}
percol = {k: [float(np.sqrt(np.mean((v[:, j][mask[:, j]] - Z[:, j][mask[:, j]]) ** 2))) for j in range(len(NUM))]
          for k, v in imps.items()}
tsave(pd.DataFrame(percol, index=[LABEL[c] for c in NUM]).round(3), "imputation_rmse_per_col")
R["m2"]["imputation_rmse"] = imp_rmse
R["m2"]["imputation_cells_hidden"] = int(mask.sum())

fig, ax = plt.subplots(figsize=(6.2, 2.3))
ks = sorted(imp_rmse, key=imp_rmse.get, reverse=True)
cols = [S.OXBLOOD if k.startswith("EM") else S.RULE for k in ks]
ax.barh(ks, [imp_rmse[k] for k in ks], color=cols, height=0.55)
for i, k in enumerate(ks):
    ax.text(imp_rmse[k] + 0.01, i, f"{imp_rmse[k]:.3f}", va="center", fontsize=8)
ax.set_xlabel("RMSE on hidden values (standard-deviation units, log scale for enzymes)")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(imp_rmse.values()) * 1.15)
S.save(fig, FIG / "f03_imputation.png")

# final EDA imputation: EM on the log scale, back-transformed; gender by mode
Xem, mu_em, Sig_em, n_it = em_mvn(L.values)
R["m2"]["em_iterations"] = int(n_it)
full = pd.DataFrame(Xem, columns=NUM)
for c in SKEWED:
    full[c] = np.expm1(full[c])
full["gender"] = df.gender.fillna(df.gender.mode()[0]).values
full["male"] = (full.gender == "Male").astype(int)
full["result"] = df.result.values
R["m2"]["gender_mode"] = str(df.gender.mode()[0])

# discretisation and binning
full["age_band"] = pd.cut(full.age, [0, 17, 39, 59, 120], labels=["Under 18", "18 to 39", "40 to 59", "60 and over"])
full["tb_band"] = pd.cut(full.tb, [0, 1.2, 3.0, 1e6], labels=["Normal (1.2 or less)", "Raised (1.2 to 3)", "High (over 3)"])
full["alb_eqwidth"] = pd.cut(full.alb, 4)
full["alb_eqfreq"] = pd.qcut(full.alb, 4)
bins_tab = {
    "age_band": full.groupby("age_band", observed=True).result.agg(["size", "mean"]),
    "tb_band": full.groupby("tb_band", observed=True).result.agg(["size", "mean"]),
    "alb_eqwidth": full.groupby("alb_eqwidth", observed=True).result.agg(["size", "mean"]),
    "alb_eqfreq": full.groupby("alb_eqfreq", observed=True).result.agg(["size", "mean"]),
}
for k, v in bins_tab.items():
    v.index = v.index.astype(str)
    tsave(v.round(4), f"bins_{k}")
R["m2"]["bins"] = {k: v.round(4).to_dict(orient="index") for k, v in bins_tab.items()}

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)
for ax, (k, title) in zip(axes, [("age_band", "Age band"), ("tb_band", "Total bilirubin band"),
                                  ("alb_eqfreq", "Albumin quartile (equal frequency)")]):
    v = bins_tab[k]
    names = [s.replace(" (", "\n(") if "(" in s and k != "alb_eqfreq" else
             (s if k == "alb_eqfreq" else s.replace(" ", "\n", 1)) for s in v.index]
    ax.bar(names, v["mean"] * 100, color=S.OXBLOOD, width=0.6)
    for i, (n_, m_) in enumerate(zip(v["size"], v["mean"])):
        ax.text(i, m_ * 100 + 2, f"{m_:.0%}", ha="center", fontsize=7.5)
    ax.set_title(title)
    ax.tick_params(axis="x", labelsize=6.8)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Disease rate (%)")
axes[0].set_ylim(0, 110)
S.save(fig, FIG / "f04_binning.png")

# ---------------------------------------------------------------- Module 3 and 4: summaries, univariate to multivariate
obs = df.copy()   # observed values only (no imputation) for descriptive statistics
summ = pd.DataFrame({
    "n": obs[NUM].count(), "mean": obs[NUM].mean(), "sd": obs[NUM].std(), "median": obs[NUM].median(),
    "q1": obs[NUM].quantile(.25), "q3": obs[NUM].quantile(.75), "min": obs[NUM].min(), "max": obs[NUM].max(),
    "cv": obs[NUM].std() / obs[NUM].mean(), "skew": obs[NUM].skew(), "kurtosis": obs[NUM].kurt()})
summ["iqr"] = summ.q3 - summ.q1
tsave(summ.round(3), "summary")
R["m4_summary"] = summ.round(3).to_dict(orient="index")

skew_after = log_frame(obs).skew()
R["m3_skew"] = {c: {"before": float(summ.loc[c, "skew"]), "after": float(skew_after[c])} for c in NUM}

# IQR outliers per variable (observed values; on the log scale for enzymes)
lf = log_frame(obs)
iqr_flags = {}
for c in NUM:
    q1, q3 = lf[c].quantile([.25, .75])
    i = q3 - q1
    iqr_flags[c] = {"raw_scale": int(((obs[c] < summ.loc[c, "q1"] - 1.5 * summ.loc[c, "iqr"]) |
                                      (obs[c] > summ.loc[c, "q3"] + 1.5 * summ.loc[c, "iqr"])).sum()),
                    "log_scale": int(((lf[c] < q1 - 1.5 * i) | (lf[c] > q3 + 1.5 * i)).sum())}
R["m3_iqr_outliers"] = iqr_flags

# univariate distributions by class
fig, axes = plt.subplots(3, 3, figsize=(7.4, 6.2))
for ax, c in zip(axes.ravel(), NUM):
    v = obs[[c, "result"]].dropna()
    logx = c in SKEWED
    bins = np.logspace(np.log10(max(v[c].min(), 0.05)), np.log10(v[c].max()), 40) if logx else 40
    for cls in (0, 1):
        ax.hist(v.loc[v.result == cls, c], bins=bins, density=True, histtype="stepfilled", alpha=0.45,
                color=S.CLASS_COLORS[cls], label=S.CLASS_NAMES[cls])
        ax.hist(v.loc[v.result == cls, c], bins=bins, density=True, histtype="step", lw=0.9,
                color=S.CLASS_COLORS[cls])
    if logx:
        ax.set_xscale("log")
    ax.set_title(f"{LABEL[c]} ({UNIT[c]}){'  log axis' if logx else ''}", fontsize=8.5)
    ax.set_yticks([])
    ax.grid(visible=False)
axes[0, 0].legend(loc="upper right", fontsize=7)
fig.tight_layout()
S.save(fig, FIG / "f05_distributions.png")

# skewness before and after (dot chart)
fig, ax = plt.subplots(figsize=(5.6, 2.8))
order = summ["skew"].sort_values().index
yy = np.arange(len(order))
ax.hlines(yy, [R["m3_skew"][c]["after"] for c in order], [R["m3_skew"][c]["before"] for c in order], color=S.RULE, lw=2)
ax.plot([R["m3_skew"][c]["before"] for c in order], yy, "o", color=S.OXBLOOD, label="raw")
ax.plot([R["m3_skew"][c]["after"] for c in order], yy, "o", color=S.TEAL, label="after log(1 + x) on the five enzyme and bilirubin columns")
ax.axvline(0, color=S.MUTED, lw=0.8)
ax.set_yticks(yy, [LABEL[c] for c in order])
ax.set_xlabel("Sample skewness")
ax.legend(loc="lower right", fontsize=7)
ax.grid(axis="y", visible=False)
S.save(fig, FIG / "f06_skew.png")

# bivariate: each variable against the outcome
mw = []
for c in NUM:
    a = obs.loc[obs.result == 1, c].dropna()
    b = obs.loc[obs.result == 0, c].dropna()
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    auc = u / (len(a) * len(b))
    mw.append({"variable": LABEL[c], "median_disease": a.median(), "median_no_disease": b.median(),
               "U": u, "p": p, "auc": auc, "rank_biserial": 2 * auc - 1})
mw = pd.DataFrame(mw).set_index("variable")
tsave(mw, "mann_whitney")
R["m3_mannwhitney"] = mw.round(6).to_dict(orient="index")

fig, axes = plt.subplots(1, 9, figsize=(7.6, 2.9))
for ax, c in zip(axes, NUM):
    data = [obs.loc[obs.result == k, c].dropna() for k in (1, 0)]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=1.5, markeredgecolor=S.MUTED, alpha=0.3),
                    medianprops=dict(color="white", lw=1.4), whiskerprops=dict(color=S.MUTED),
                    capprops=dict(color=S.MUTED))
    for patch, k in zip(bp["boxes"], (1, 0)):
        patch.set_facecolor(S.CLASS_COLORS[k])
        patch.set_edgecolor(S.CLASS_COLORS[k])
    if c in SKEWED:
        ax.set_yscale("log")
    ax.set_xticks([1, 2], ["D", "N"])
    ax.set_title(LABEL[c].replace(" ", "\n", 1), fontsize=7.2)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="x", visible=False)
fig.tight_layout(w_pad=0.4)
S.save(fig, FIG / "f07_boxplots.png")

# correlation, Pearson on log scale and Spearman
corr_p = log_frame(obs).corr(method="pearson")
corr_s = obs[NUM].corr(method="spearman")
tsave(corr_p.round(3), "corr_pearson_log")
tsave(corr_s.round(3), "corr_spearman")
R["m3_corr"] = {"pearson_log": corr_p.round(3).to_dict(), "spearman": corr_s.round(3).to_dict(),
                "pearson_raw_tb_db": float(obs[["tb", "db"]].corr().iloc[0, 1]),
                "pearson_raw_sgpt_sgot": float(obs[["sgpt", "sgot"]].corr().iloc[0, 1])}
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
short = ["Age", "TBil", "DBil", "ALP", "ALT", "AST", "TP", "Alb", "A/G"]
for ax, cm, title in zip(axes, [corr_p, corr_s], ["Pearson, enzymes and bilirubin logged", "Spearman rank"]):
    im = ax.imshow(cm.values, cmap=S.DIV_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(9), short, rotation=90)
    ax.set_yticks(range(9), short)
    ax.grid(visible=False)
    for i in range(9):
        for j in range(9):
            if i != j:
                ax.text(j, i, f"{cm.values[i, j]:.2f}".replace("0.", "."), ha="center", va="center", fontsize=6,
                        color="white" if abs(cm.values[i, j]) > 0.6 else S.INK)
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)
fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
S.save(fig, FIG / "f08_correlation.png")

# scatter plots
sub = full.sample(4000, random_state=SEED)
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
for ax, (x, y_) in zip(axes, [("tb", "db"), ("sgpt", "sgot"), ("tp", "alb")]):
    for k in (0, 1):
        s = sub[sub.result == k]
        ax.scatter(s[x], s[y_], s=3, alpha=0.35, color=S.CLASS_COLORS[k], label=S.CLASS_NAMES[k], lw=0)
    if x in SKEWED:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(f"{LABEL[x]} ({UNIT[x]})")
    ax.set_ylabel(f"{LABEL[y_]} ({UNIT[y_]})")
axes[0].legend(markerscale=4, loc="upper left")
fig.tight_layout()
S.save(fig, FIG / "f09_scatter.png")

# pair plot of four variables
four = ["tb", "sgot", "alb", "age"]
sub2 = full.sample(1500, random_state=1)
fig, axes = plt.subplots(4, 4, figsize=(7.0, 6.6))
for i, a in enumerate(four):
    for j, b in enumerate(four):
        ax = axes[i, j]
        xa = np.log10(sub2[b]) if b in SKEWED else sub2[b]
        ya = np.log10(sub2[a]) if a in SKEWED else sub2[a]
        if i == j:
            for k in (0, 1):
                ax.hist(xa[sub2.result == k], bins=25, density=True, alpha=0.5, color=S.CLASS_COLORS[k])
            ax.set_yticks([])
        else:
            for k in (0, 1):
                m = sub2.result == k
                ax.scatter(xa[m], ya[m], s=2, alpha=0.4, color=S.CLASS_COLORS[k], lw=0)
        if i == 3:
            ax.set_xlabel(LABEL[b] + (" (log10)" if b in SKEWED else ""), fontsize=7.5)
        else:
            ax.set_xticklabels([])
        if j == 0 and i != 0:
            ax.set_ylabel(LABEL[a] + (" (log10)" if a in SKEWED else ""), fontsize=7.5)
        elif j != 0:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=6.5)
fig.tight_layout(h_pad=0.3, w_pad=0.3)
S.save(fig, FIG / "f10_pairs.png")

# contingency tables
def contingency(row, name):
    tab = pd.crosstab(full[row], full.result)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    n = tab.values.sum()
    v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
    tab.columns = ["No disease", "Liver disease"]
    tab["Disease rate"] = tab["Liver disease"] / tab.sum(axis=1)
    tab.index = tab.index.astype(str)
    tsave(tab.round(4), f"contingency_{name}")
    return {"table": tab.round(4).to_dict(orient="index"), "chi2": float(chi2), "p": float(p), "dof": int(dof),
            "cramers_v": float(v)}


R["m4_contingency"] = {"gender": contingency("gender", "gender"), "age_band": contingency("age_band", "age"),
                       "tb_band": contingency("tb_band", "tb")}
# gender in observed rows only, to check the mode imputation did not change the story
tabg = pd.crosstab(df.gender.dropna(), df.loc[df.gender.notna(), "result"])
R["m4_contingency"]["gender_observed_only_p"] = float(stats.chi2_contingency(tabg)[1])

# n-D: age band x gender x bilirubin band
nd = full.pivot_table(index=["age_band", "gender"], columns="tb_band", values="result", aggfunc="mean", observed=True)
ndn = full.pivot_table(index=["age_band", "gender"], columns="tb_band", values="result", aggfunc="size", observed=True)
tsave(nd.round(3), "nd_rate")
tsave(ndn, "nd_count")
fig, ax = plt.subplots(figsize=(5.8, 3.4))
im = ax.imshow(nd.values, cmap=S.SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(nd.shape[1]), [s.replace(" (", "\n(") for s in nd.columns.astype(str)])
ax.set_yticks(range(nd.shape[0]), [f"{a}, {g.lower()}" for a, g in nd.index])
ax.grid(visible=False)
for i in range(nd.shape[0]):
    for j in range(nd.shape[1]):
        val, cnt = nd.values[i, j], ndn.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.0%}  (n={int(cnt):,})", ha="center", va="center", fontsize=7,
                    color="white" if val > 0.6 else S.INK)
for s in ax.spines.values():
    s.set_visible(False)
fig.colorbar(im, ax=ax, shrink=0.8, label="Disease rate")
S.save(fig, FIG / "f11_nd_heatmap.png")

# dot chart: standardised median difference
fig, ax = plt.subplots(figsize=(5.8, 2.9))
mw_sorted = mw.sort_values("rank_biserial")
yy = np.arange(len(mw_sorted))
ax.hlines(yy, 0, mw_sorted.rank_biserial, color=S.RULE, lw=1.5)
ax.plot(mw_sorted.rank_biserial, yy, "o", color=S.OXBLOOD, markersize=6)
for i, (r_, p_) in enumerate(zip(mw_sorted.rank_biserial, mw_sorted.p)):
    ax.text(r_ + (0.012 if r_ >= 0 else -0.012), i, f"{r_:+.2f}", va="center", ha="left" if r_ >= 0 else "right", fontsize=7.5)
ax.axvline(0, color=S.MUTED, lw=0.8)
ax.set_yticks(yy, mw_sorted.index)
ax.set_xlabel("Rank-biserial correlation with disease (positive: higher in patients)")
ax.set_xlim(-0.45, 0.6)
ax.grid(axis="y", visible=False)
S.save(fig, FIG / "f12_dotchart.png")

# bar plot with 95% Wilson intervals: age band x gender
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


g = full.groupby(["age_band", "gender"], observed=True).result.agg(["sum", "size"]).reset_index()
fig, ax = plt.subplots(figsize=(5.8, 2.6))
bands = g.age_band.cat.categories
w = 0.36
for k, (sex, col) in enumerate([("Female", S.TEAL), ("Male", S.OXBLOOD)]):
    gg = g[g.gender == sex].set_index("age_band").reindex(bands)
    rate = gg["sum"] / gg["size"]
    lo, hi = wilson(gg["sum"].values, gg["size"].values)
    x = np.arange(len(bands)) + (k - 0.5) * w
    ax.bar(x, rate * 100, width=w, color=col, label=sex)
    ax.errorbar(x, rate * 100, yerr=[(rate - lo) * 100, (hi - rate) * 100], fmt="none", ecolor=S.INK, lw=0.8, capsize=2)
ax.set_xticks(range(len(bands)), bands)
ax.set_ylabel("Disease rate (%)")
ax.set_ylim(0, 100)
ax.legend(ncol=2, loc="upper left")
ax.grid(axis="x", visible=False)
S.save(fig, FIG / "f13_bars.png")

# ---------------------------------------------------------------- Module 5: clustering
Lf = log_frame(full)
scaler_all = StandardScaler().fit(Lf)
Zall = scaler_all.transform(Lf)
y_all = full.result.values

bic = []
for k in range(1, 11):
    gm = GaussianMixture(k, covariance_type="full", random_state=SEED, n_init=2).fit(Zall)
    bic.append(gm.bic(Zall))
k_bic = int(np.argmin(bic) + 1)
R["m5"] = {"bic": [float(b) for b in bic], "k_bic": k_bic}
fig, ax = plt.subplots(figsize=(5.2, 2.3))
ax.plot(range(1, 11), np.array(bic) / 1000, "-o", color=S.SLATE, markersize=4)
ax.plot(k_bic, bic[k_bic - 1] / 1000, "o", color=S.OXBLOOD, markersize=8)
ax.set_xlabel("Number of mixture components")
ax.set_ylabel("BIC (thousands)")
ax.set_xticks(range(1, 11))
S.save(fig, FIG / "f14_gmm_bic.png")

K = 4
idx_sub = rng.choice(len(Zall), 2000, replace=False)
Zs, ys = Zall[idx_sub], y_all[idx_sub]
lab_sets = {}
lab_sets["K-means"] = KMeans(K, n_init=10, random_state=SEED).fit_predict(Zs)
gm4 = GaussianMixture(K, covariance_type="full", random_state=SEED, n_init=3).fit(Zall)
lab_sets["Gaussian mixture (EM)"] = gm4.predict(Zs)
Lw = linkage(Zs, method="ward")
lab_sets["Agglomerative, Ward"] = fcluster(Lw, K, criterion="maxclust") - 1
lab_sets["Spectral"] = SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=15, random_state=SEED,
                                          assign_labels="cluster_qr").fit_predict(Zs)
Dfull = squareform(pdist(Zs))
mst = minimum_spanning_tree(Dfull).toarray()
edges = np.argwhere(mst > 0)
w_ = mst[mst > 0]
cut = np.argsort(w_)[-(K - 1):]
mst_cut = mst.copy()
for e in edges[cut]:
    mst_cut[e[0], e[1]] = 0
_, mst_lab = connected_components(mst_cut, directed=False)
lab_sets["Minimum spanning tree"] = mst_lab

clus = []
for name, labs in lab_sets.items():
    sizes = np.bincount(labs)
    sil = silhouette_score(Zs, labs) if len(np.unique(labs)) > 1 else np.nan
    clus.append({"method": name, "sizes": "/".join(str(s) for s in sorted(sizes, reverse=True)),
                 "silhouette": sil, "ari_vs_outcome": adjusted_rand_score(ys, labs),
                 "max_cluster_disease_rate": max(ys[labs == c].mean() for c in np.unique(labs)),
                 "min_cluster_disease_rate": min(ys[labs == c].mean() for c in np.unique(labs))})
clus = pd.DataFrame(clus).set_index("method")
tsave(clus.round(4), "clustering")
R["m5"]["methods"] = clus.round(4).to_dict(orient="index")
R["m5"]["mst_longest_edges"] = [float(x) for x in np.sort(w_)[-5:]]
R["m5"]["mst_median_edge"] = float(np.median(w_))

# profile the GMM clusters on the full data
gm_lab_all = gm4.predict(Zall)
prof = full.assign(cluster=gm_lab_all).groupby("cluster").agg(
    n=("result", "size"), disease_rate=("result", "mean"), age=("age", "median"), tb=("tb", "median"),
    sgpt=("sgpt", "median"), sgot=("sgot", "median"), alkphos=("alkphos", "median"), alb=("alb", "median"),
    ag=("ag", "median"))
prof = prof.sort_values("disease_rate", ascending=False)
tsave(prof.round(3), "gmm_profile")
R["m5"]["gmm_profile"] = prof.round(3).to_dict(orient="index")

pca2 = PCA(2, random_state=SEED).fit(Zall)
P2 = pca2.transform(Zs)
fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.0))
pal = [S.OXBLOOD, S.TEAL, S.SAND, S.SLATE]
for ax, (name, labs) in zip(axes.ravel(), list(lab_sets.items()) + [("Outcome (for reference)", ys)]):
    for c in np.unique(labs):
        m = labs == c
        col = S.CLASS_COLORS[c] if name.startswith("Outcome") else pal[c % 4]
        ax.scatter(P2[m, 0], P2[m, 1], s=3, color=col, alpha=0.6, lw=0)
    ax.set_title(name, fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
S.save(fig, FIG / "f16_clusters.png")

fig, ax = plt.subplots(figsize=(7.4, 2.8))
dendrogram(Lw, truncate_mode="lastp", p=30, ax=ax, color_threshold=Lw[-(K - 1), 2],
           above_threshold_color=S.MUTED, leaf_font_size=6.5)
ax.set_ylabel("Ward linkage distance")
ax.grid(axis="x", visible=False)
ax.axhline(Lw[-(K - 1), 2] - 1e-6, color=S.OXBLOOD, lw=0.8, ls="--")
S.save(fig, FIG / "f15_dendrogram.png")

# outlier detection through clustering
dens = gm4.score_samples(Zall)
out_gmm = dens < np.quantile(dens, 0.01)
km_all = KMeans(K, n_init=10, random_state=SEED).fit(Zall)
dist_c = np.linalg.norm(Zall - km_all.cluster_centers_[km_all.labels_], axis=1)
out_km = dist_c > np.quantile(dist_c, 0.99)
db = DBSCAN(eps=1.2, min_samples=10).fit(Zall)
out_db = db.labels_ == -1
mst_out_sub = np.zeros(len(Zall), bool)
small = [c for c in np.unique(mst_lab) if (mst_lab == c).sum() < 20]
mst_out_sub[idx_sub[np.isin(mst_lab, small)]] = True


def jac(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


R["m5"]["outliers"] = {
    "gmm_low_density": int(out_gmm.sum()), "kmeans_far": int(out_km.sum()), "dbscan_noise": int(out_db.sum()),
    "jaccard_gmm_km": jac(out_gmm, out_km), "jaccard_gmm_db": jac(out_gmm, out_db),
    "disease_rate_gmm_outliers": float(y_all[out_gmm].mean()),
    "disease_rate_rest": float(y_all[~out_gmm].mean()),
    "median_tb_outliers": float(full.tb[out_gmm].median()), "median_sgot_outliers": float(full.sgot[out_gmm].median()),
    "mst_isolated_points": int(mst_out_sub.sum()),
}
Pall = pca2.transform(Zall)
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.scatter(Pall[~out_gmm, 0], Pall[~out_gmm, 1], s=2, color=S.RULE, lw=0, label="typical")
ax.scatter(Pall[out_gmm & ~out_km, 0], Pall[out_gmm & ~out_km, 1], s=7, color=S.SAND, lw=0, label="low GMM density only")
ax.scatter(Pall[out_gmm & out_km, 0], Pall[out_gmm & out_km, 1], s=9, color=S.OXBLOOD, lw=0,
           label="low density and far from K-means centre")
ax.scatter(Pall[out_km & ~out_gmm, 0], Pall[out_km & ~out_gmm, 1], s=7, color=S.TEAL, lw=0, label="far from centre only")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(markerscale=2, fontsize=7, loc="upper right")
S.save(fig, FIG / "f17_outliers.png")

# ---------------------------------------------------------------- Module 6: dimensionality reduction
pca = PCA().fit(Zall)
ev = pca.explained_variance_ratio_
R["m6"] = {"pca_var": [float(v) for v in ev], "pca_cum": [float(v) for v in np.cumsum(ev)],
           "n90": int(np.searchsorted(np.cumsum(ev), 0.90) + 1), "n95": int(np.searchsorted(np.cumsum(ev), 0.95) + 1),
           "kaiser": int((pca.explained_variance_ > 1).sum())}
U, sv, Vt = np.linalg.svd(Zall - Zall.mean(0), full_matrices=False)
R["m6"]["svd_eig_maxdiff"] = float(np.abs(sv ** 2 / (len(Zall) - 1) - pca.explained_variance_).max())
R["m6"]["svd_singular_values"] = [float(s) for s in sv]
load = pd.DataFrame(pca.components_[:4].T, index=[LABEL[c] for c in NUM], columns=["PC1", "PC2", "PC3", "PC4"])
tsave(load.round(3), "pca_loadings")
R["m6"]["loadings"] = load.round(3).to_dict()

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.1, 1]})
axes[0].bar(range(1, 10), ev * 100, color=S.SLATE, width=0.6)
axes[0].plot(range(1, 10), np.cumsum(ev) * 100, "-o", color=S.OXBLOOD, markersize=4)
axes[0].axhline(90, color=S.MUTED, ls="--", lw=0.7)
axes[0].set_xlabel("Principal component")
axes[0].set_ylabel("Variance explained (%)")
axes[0].set_xticks(range(1, 10))
axes[0].grid(axis="x", visible=False)
im = axes[1].imshow(load.values, cmap=S.DIV_CMAP, vmin=-0.7, vmax=0.7, aspect="auto")
axes[1].set_xticks(range(4), load.columns)
axes[1].set_yticks(range(9), load.index, fontsize=7.5)
axes[1].grid(visible=False)
for i in range(9):
    for j in range(4):
        axes[1].text(j, i, f"{load.values[i, j]:+.2f}", ha="center", va="center", fontsize=6.5,
                     color="white" if abs(load.values[i, j]) > 0.45 else S.INK)
for s in axes[1].spines.values():
    s.set_visible(False)
fig.tight_layout()
S.save(fig, FIG / "f18_pca.png")

# biplot
fig, ax = plt.subplots(figsize=(5.8, 4.0))
Pb = pca.transform(Zs)[:, :2]
for k in (0, 1):
    ax.scatter(Pb[ys == k, 0], Pb[ys == k, 1], s=3, alpha=0.4, color=S.CLASS_COLORS[k], lw=0, label=S.CLASS_NAMES[k])
sc = 4.5
nudge = {"tb": (0.15, -0.05, "Total and direct bilirubin"), "db": None, "sgpt": (-0.15, 0.22, None),
         "sgot": (0.35, -0.08, None), "alkphos": (0.25, -0.18, None), "tp": (0.35, 0.12, None),
         "alb": (-0.35, 0.18, None), "ag": (-0.45, -0.05, None), "age": (0.25, -0.35, "Age (near zero)")}
for i, c in enumerate(NUM):
    x_, y_ = pca.components_[0, i] * sc, pca.components_[1, i] * sc
    ax.arrow(0, 0, x_, y_, color=S.INK, width=0.012, head_width=0.12, length_includes_head=True, zorder=3)
    if nudge[c] is None:
        continue
    dx, dy, txt = nudge[c]
    ax.text(x_ + dx, y_ + dy, txt or LABEL[c], fontsize=7.5, ha="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)")
ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
ax.legend(markerscale=4, loc="lower left")
S.save(fig, FIG / "f19_biplot.png")

fa = FactorAnalysis(3, rotation="varimax", random_state=SEED).fit(Zall)
fal = pd.DataFrame(fa.components_.T, index=[LABEL[c] for c in NUM], columns=["Factor 1", "Factor 2", "Factor 3"])
fal["Uniqueness"] = fa.noise_variance_
tsave(fal.round(3), "factor_loadings")
R["m6"]["fa"] = fal.round(3).to_dict()
fig, ax = plt.subplots(figsize=(4.8, 3.1))
im = ax.imshow(fal.values[:, :3], cmap=S.DIV_CMAP, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(3), fal.columns[:3])
ax.set_yticks(range(9), fal.index)
ax.grid(visible=False)
for i in range(9):
    for j in range(3):
        ax.text(j, i, f"{fal.values[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(fal.values[i, j]) > 0.6 else S.INK)
for s in ax.spines.values():
    s.set_visible(False)
S.save(fig, FIG / "f20_factors.png")


def mle_id(X, k):
    """Levina and Bickel (2004) maximum-likelihood estimate of intrinsic dimension."""
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    d = d[:, 1:]
    keep = (d > 1e-9).all(axis=1)
    d = d[keep]
    m = (k - 1) / np.log(d[:, -1][:, None] / d[:, :-1]).sum(axis=1)
    return float(np.mean(m))


Zu = np.unique(np.round(Zall, 6), axis=0)
Zid = Zu[rng.choice(len(Zu), 4000, replace=False)]
R["m6"]["mle_id"] = {k: mle_id(Zid, k) for k in (10, 15, 20)}

# non-linear embeddings
idx_e = idx_sub[:1500]
Ze, ye = Zall[idx_e], y_all[idx_e]
emb = {}
emb["Metric MDS"] = MDS(n_components=2, random_state=SEED, n_init=1, max_iter=200).fit_transform(Ze[:700])
emb["Isomap (12 neighbours)"] = Isomap(n_neighbors=12, n_components=2).fit_transform(Ze)
emb["t-SNE (perplexity 30)"] = TSNE(2, perplexity=30, random_state=SEED, init="pca").fit_transform(Ze)
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
for ax, (name, E) in zip(axes, emb.items()):
    yy_ = ye[: len(E)]
    for k in (0, 1):
        ax.scatter(E[yy_ == k, 0], E[yy_ == k, 1], s=3, alpha=0.55, color=S.CLASS_COLORS[k], lw=0, label=S.CLASS_NAMES[k])
    ax.set_title(name, fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])
axes[0].legend(markerscale=4, fontsize=7, loc="lower left")
fig.tight_layout()
S.save(fig, FIG / "f21_embeddings.png")
# how well does each 2-D map keep the classes apart? 5-NN accuracy inside the embedding
from sklearn.model_selection import cross_val_score
R["m6"]["embed_knn_acc"] = {name: float(cross_val_score(KNeighborsClassifier(5), E, ye[: len(E)], cv=5).mean())
                            for name, E in emb.items()}
R["m6"]["embed_knn_acc"]["PCA (2 components)"] = float(
    cross_val_score(KNeighborsClassifier(5), pca.transform(Ze)[:, :2], ye, cv=5).mean())


def train_som(X, rows=10, cols=10, epochs=6, seed=SEED):
    r = np.random.default_rng(seed)
    W = X[r.choice(len(X), rows * cols, replace=False)].copy()
    grid = np.array([(i, j) for i in range(rows) for j in range(cols)], float)
    T = epochs * len(X)
    t = 0
    for ep in range(epochs):
        for x in X[r.permutation(len(X))]:
            frac = t / T
            lr = 0.5 * (1 - frac) + 0.01
            sig = max(rows / 2 * (1 - frac), 0.6)
            bmu = np.argmin(((W - x) ** 2).sum(1))
            h = np.exp(-((grid - grid[bmu]) ** 2).sum(1) / (2 * sig * sig))
            W += lr * h[:, None] * (x - W)
            t += 1
    return W, grid


Xsom_idx = rng.choice(len(Zall), 4000, replace=False)
W, grid = train_som(Zall[Xsom_idx])
bmu_all = np.argmin(((Zall[:, None, :] - W[None]) ** 2).sum(2), axis=1)
qe = float(np.mean(np.linalg.norm(Zall - W[bmu_all], axis=1)))
umat = np.zeros((10, 10))
for n_ in range(100):
    i, j = grid[n_].astype(int)
    nb = [m for m in range(100) if abs(grid[m] - grid[n_]).sum() == 1]
    umat[i, j] = np.mean([np.linalg.norm(W[n_] - W[m]) for m in nb])
hits = np.bincount(bmu_all, minlength=100).reshape(10, 10)
rate = np.array([y_all[bmu_all == n_].mean() if (bmu_all == n_).any() else np.nan for n_ in range(100)]).reshape(10, 10)
R["m6"]["som"] = {"quantisation_error": qe, "empty_nodes": int((hits == 0).sum()),
                  "node_rate_min": float(np.nanmin(rate)), "node_rate_max": float(np.nanmax(rate)),
                  "nodes_rate_below_50": int((rate < 0.5).sum())}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
im0 = axes[0].imshow(umat, cmap="Greys")
axes[0].set_title("U-matrix (dark = cluster boundary)")
im1 = axes[1].imshow(rate, cmap=S.SEQ_CMAP, vmin=0, vmax=1)
axes[1].set_title("Disease rate of records mapped to each node")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(visible=False)
fig.colorbar(im0, ax=axes[0], shrink=0.75)
fig.colorbar(im1, ax=axes[1], shrink=0.75)
S.save(fig, FIG / "f22_som.png")
print(f"EDA done in {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------- Module 7a: linear regression
reg = obs[NUM + ["gender"]].dropna().copy()
reg["male"] = (reg.gender == "Male").astype(int)
for c in SKEWED:
    reg["log_" + c] = np.log(reg[c])
yreg = reg["log_sgot"]
xcols = ["log_sgpt", "log_alkphos", "log_tb", "age", "alb", "male"]
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(reg[xcols], yreg, test_size=0.2, random_state=SEED)
ols = sm.OLS(ytr_r, sm.add_constant(Xtr_r)).fit()
pred_r = ols.predict(sm.add_constant(Xte_r))
res_r = yte_r - pred_r
R["m7_reg"] = {
    "n_train": int(len(Xtr_r)), "n_test": int(len(Xte_r)),
    "r2_train": float(ols.rsquared), "adj_r2_train": float(ols.rsquared_adj),
    "r2_test": float(1 - (res_r ** 2).sum() / ((yte_r - yte_r.mean()) ** 2).sum()),
    "rmse_test_log": float(np.sqrt((res_r ** 2).mean())), "mae_test_log": float(res_r.abs().mean()),
    "median_ape_test": float(np.median(np.abs(np.exp(pred_r) - np.exp(yte_r)) / np.exp(yte_r))),
    "within_25pct": float(np.mean(np.abs(np.exp(pred_r) - np.exp(yte_r)) / np.exp(yte_r) <= 0.25)),
    "coef": {k: {"b": float(ols.params[k]), "se": float(ols.bse[k]), "t": float(ols.tvalues[k]), "p": float(ols.pvalues[k])}
             for k in ols.params.index},
    "simple_r2_sgpt_only": float(sm.OLS(ytr_r, sm.add_constant(Xtr_r[["log_sgpt"]])).fit().rsquared),
}
coef_tab = pd.DataFrame(R["m7_reg"]["coef"]).T
tsave(coef_tab.round(5), "regression_coef")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
axes[0].scatter(np.exp(yte_r), np.exp(pred_r), s=3, alpha=0.35, color=S.SLATE, lw=0)
lim = [8, 5000]
axes[0].plot(lim, lim, color=S.OXBLOOD, lw=1)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlim(lim)
axes[0].set_ylim(lim)
axes[0].set_xlabel("Measured AST (IU/L)")
axes[0].set_ylabel("Predicted AST (IU/L)")
axes[0].set_title(f"Test set, R² = {R['m7_reg']['r2_test']:.2f}")
axes[1].scatter(pred_r, res_r, s=3, alpha=0.35, color=S.SLATE, lw=0)
axes[1].axhline(0, color=S.OXBLOOD, lw=1)
axes[1].set_xlabel("Fitted log AST")
axes[1].set_ylabel("Residual (log units)")
axes[1].set_title("Residuals against fitted values")
fig.tight_layout()
S.save(fig, FIG / "f23_regression.png")

# ---------------------------------------------------------------- Module 7b: classification under two split protocols
X = df[NUM].copy()
X["male"] = df.gender.map({"Male": 1.0, "Female": 0.0})
yb = df.result.values

# assign every row to its nearest record in the original 583-patient release ("source family")
Lo = log_frame(orig)
med = Lo.median()
mu_o, sd_o = Lo.mean(), Lo.std()
nn_o = NearestNeighbors(n_neighbors=1).fit(((Lo.fillna(med) - mu_o) / sd_o).values)
d_anchor, anchor = nn_o.kneighbors(((log_frame(df).fillna(med) - mu_o) / sd_o).values)
groups = anchor[:, 0]
R["m7_groups"] = {"n_groups": int(len(np.unique(groups))), "median_family": float(pd.Series(groups).value_counts().median()),
                  "max_family": int(pd.Series(groups).value_counts().max()),
                  "label_agrees_with_anchor": float((orig.result.values[groups] == yb).mean()),
                  "anchor_dist_median": float(np.median(d_anchor))}


def make_pre():
    num_pipe = Pipeline([("log", FunctionTransformer(lambda a: np.c_[a[:, :1], np.log1p(a[:, 1:6]), a[:, 6:]],
                                                     feature_names_out="one-to-one")),
                         ("impute", IterativeImputer(random_state=SEED, max_iter=15)),
                         ("scale", StandardScaler())])
    sex_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    return ColumnTransformer([("num", num_pipe, list(range(9))), ("sex", sex_pipe, [9])])


ratio = (yb == 0).sum() / (yb == 1).sum()
MODELS = {
    "Logistic regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "Gaussian naive Bayes": GaussianNB(),
    "k-nearest neighbours (k=15)": KNeighborsClassifier(15),
    "Decision tree": DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
    "Random forest": RandomForestClassifier(400, class_weight="balanced", n_jobs=-1, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=ratio, eval_metric="logloss", n_jobs=-1, random_state=SEED),
    "SVM (RBF kernel)": SVC(C=1.0, gamma="scale", class_weight="balanced", random_state=SEED),
}
SHORT = {"Logistic regression": "LR", "Gaussian naive Bayes": "NB", "k-nearest neighbours (k=15)": "KNN",
         "Decision tree": "DT", "Random forest": "RF", "XGBoost": "XGB", "SVM (RBF kernel)": "SVM"}


def score(model, Xs):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xs)[:, 1]
    return model.decision_function(Xs)


def boot_ci(y, s, pred, n=500):
    r = np.random.default_rng(7)
    aucs, accs = [], []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        aucs.append(roc_auc_score(y[i], s[i]))
        accs.append(accuracy_score(y[i], pred[i]))
    return np.percentile(aucs, [2.5, 97.5]), np.percentile(accs, [2.5, 97.5])


protocols = {}
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
g_tr, g_te = next(gss.split(X, yb, groups))
r_tr, r_te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=yb, random_state=SEED)
protocols["A"] = {"name": "Random stratified 80/20 split", "tr": r_tr, "te": r_te,
                  "cv": StratifiedKFold(5, shuffle=True, random_state=SEED), "groups": None}
protocols["B"] = {"name": "Source-family grouped 80/20 split", "tr": g_tr, "te": g_te,
                  "cv": StratifiedGroupKFold(5, shuffle=True, random_state=SEED), "groups": groups}
for k_, P_ in protocols.items():
    P_.update(Xtr=X.values[P_["tr"]], Xte=X.values[P_["te"]], ytr=yb[P_["tr"]], yte=yb[P_["te"]],
              sex_te=df.gender.values[P_["te"]], shared=len(set(groups[P_["tr"]]) & set(groups[P_["te"]])),
              groups_tr=groups[P_["tr"]] if P_["groups"] is not None else None)
# protocol C: only the 570 unique patients of the original release
Xo = orig_u[NUM].copy()
Xo["male"] = orig_u.gender.map({"Male": 1.0, "Female": 0.0})
yo = orig_u.result.values
o_tr, o_te = train_test_split(np.arange(len(Xo)), test_size=0.2, stratify=yo, random_state=SEED)
protocols["C"] = {"name": "Original 570 patients, stratified 80/20 split", "Xtr": Xo.values[o_tr], "Xte": Xo.values[o_te],
                  "ytr": yo[o_tr], "yte": yo[o_te], "sex_te": orig_u.gender.values[o_te], "shared": 0, "groups_tr": None,
                  "cv": RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)}


def pool_rule(Xtr_, ytr_, Xte_):
    """No learning at all: for each column, is the test value seen only in one class of the training rows?"""
    score = np.zeros(len(Xte_))
    for j in range(len(NUM)):
        s1 = set(Xtr_[ytr_ == 1, j][~np.isnan(Xtr_[ytr_ == 1, j])])
        s0 = set(Xtr_[ytr_ == 0, j][~np.isnan(Xtr_[ytr_ == 0, j])])
        v = Xte_[:, j]
        score += np.isin(v, list(s1 - s0)).astype(int) - np.isin(v, list(s0 - s1)).astype(int)
    return score


R["m7_pool_rule"] = {}
for k_, P_ in protocols.items():
    sc_ = pool_rule(P_["Xtr"], P_["ytr"], P_["Xte"])
    pr_ = (sc_ >= 0).astype(int)
    R["m7_pool_rule"][k_] = {"acc": float(accuracy_score(P_["yte"], pr_)), "auc": float(roc_auc_score(P_["yte"], sc_)),
                             "bacc": float(balanced_accuracy_score(P_["yte"], pr_)),
                             "share_with_exclusive_value": float((sc_ != 0).mean())}

R["m7"] = {}
fitted = {}
rocs = {}
for key, P in protocols.items():
    Xtr, Xte, ytr, yte = P["Xtr"], P["Xte"], P["ytr"], P["yte"]
    shared_groups = P["shared"]
    # leakage evidence: distance from each test row to its nearest training row
    pre_leak = make_pre().fit(Xtr)
    nn_tr = NearestNeighbors(n_neighbors=1).fit(pre_leak.transform(Xtr))
    dnn, inn = nn_tr.kneighbors(pre_leak.transform(Xte))
    info = {"name": P["name"], "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "test_disease_share": float(yte.mean()), "train_disease_share": float(ytr.mean()),
            "shared_source_families": int(shared_groups),
            "nn_dist_median": float(np.median(dnn)), "nn_dist_q": [float(q) for q in np.quantile(dnn, [.1, .25, .5, .75, .9])],
            "nn_label_agreement": float((ytr[inn[:, 0]] == yte).mean()), "models": {}}
    rocs[key] = {}
    np.save(ROOT / "results" / f"nn_dist_{key}.npy", dnn[:, 0])
    for mname, base in MODELS.items():
        t1 = time.time()
        pipe = Pipeline([("pre", make_pre()), ("clf", clone(base))])
        cv = cross_validate(pipe, Xtr, ytr, groups=P["groups_tr"], cv=P["cv"],
                            scoring=["roc_auc", "accuracy", "f1", "balanced_accuracy"], n_jobs=1)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        s = score(pipe, Xte)
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
        (auc_lo, auc_hi), (acc_lo, acc_hi) = boot_ci(yte, s, pred)
        info["models"][mname] = {
            "cv_auc_mean": float(cv["test_roc_auc"].mean()), "cv_auc_sd": float(cv["test_roc_auc"].std()),
            "cv_acc_mean": float(cv["test_accuracy"].mean()), "cv_f1_mean": float(cv["test_f1"].mean()),
            "cv_bacc_mean": float(cv["test_balanced_accuracy"].mean()),
            "train_acc": float(accuracy_score(ytr, pipe.predict(Xtr))),
            "acc": float(accuracy_score(yte, pred)), "acc_ci": [float(acc_lo), float(acc_hi)],
            "precision": float(precision_score(yte, pred)), "recall": float(recall_score(yte, pred)),
            "specificity": float(tn / (tn + fp)), "f1": float(f1_score(yte, pred)),
            "bacc": float(balanced_accuracy_score(yte, pred)), "auc": float(roc_auc_score(yte, s)),
            "auc_ci": [float(auc_lo), float(auc_hi)], "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "seconds": float(time.time() - t1)}
        fpr, tpr, _ = roc_curve(yte, s)
        rocs[key][mname] = (fpr, tpr)
        fitted[(key, mname)] = (pipe, pred, s)
        print(f"{key} {mname}: acc {info['models'][mname]['acc']:.3f} auc {info['models'][mname]['auc']:.3f} "
              f"({time.time() - t1:.0f}s)", flush=True)
    best = max(info["models"], key=lambda m: info["models"][m]["cv_auc_mean"])
    info["best_by_cv_auc"] = best
    ranked = sorted(info["models"], key=lambda m: info["models"][m]["auc"], reverse=True)
    info["ranked_by_test_auc"] = ranked
    # McNemar between the two best models on the test set
    p1, p2 = fitted[(key, ranked[0])][1], fitted[(key, ranked[1])][1]
    c1, c2 = p1 == yte, p2 == yte
    tbl = [[int((c1 & c2).sum()), int((c1 & ~c2).sum())], [int((~c1 & c2).sum()), int((~c1 & ~c2).sum())]]
    mc = mcnemar(tbl, exact=False, correction=True)
    info["mcnemar"] = {"models": ranked[:2], "table": tbl, "stat": float(mc.statistic), "p": float(mc.pvalue)}
    R["m7"][key] = info
    rows = []
    for m, v in info["models"].items():
        rows.append({"model": m, **{k: v[k] for k in ["cv_auc_mean", "cv_auc_sd", "cv_acc_mean", "train_acc", "acc",
                                                       "precision", "recall", "specificity", "f1", "bacc", "auc"]}})
    tsave(pd.DataFrame(rows).set_index("model").round(4), f"models_{key}")

# confusion-matrix grids
for key in ("A", "B", "C"):
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.0))
    for ax, mname in zip(axes.ravel(), MODELS):
        cm = np.array(R["m7"][key]["models"][mname]["cm"])
        rown = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rown, cmap=S.SEQ_CMAP, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{rown[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                        color="white" if rown[i, j] > 0.55 else S.INK)
        ax.set_xticks([0, 1], ["No dis.", "Disease"], fontsize=7)
        ax.set_yticks([0, 1], ["No dis.", "Disease"], fontsize=7)
        ax.set_title(f"{SHORT[mname]}  acc {R['m7'][key]['models'][mname]['acc']:.3f}", fontsize=8)
        ax.grid(visible=False)
        for s_ in ax.spines.values():
            s_.set_visible(False)
    ax = axes.ravel()[-1]
    ax.axis("off")
    ax.text(0, 0.5, "Rows: actual class\nColumns: predicted class\nCell: count and row share\n\n"
            f"Test set: {R['m7'][key]['n_test']:,} records", fontsize=7.5, va="center", color=S.MUTED)
    fig.tight_layout()
    S.save(fig, FIG / f"f24_cm_{key}.png")

fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0), sharey=True)
for ax, key in zip(axes, ("A", "B", "C")):
    for (mname, (fpr, tpr)), col in zip(rocs[key].items(), S.MODEL_COLORS):
        ax.plot(fpr, tpr, color=col, lw=1.3, label=f"{SHORT[mname]}  {R['m7'][key]['models'][mname]['auc']:.3f}")
    ax.plot([0, 1], [0, 1], color=S.RULE, lw=0.8, ls="--")
    ax.set_title(f"Protocol {key}", fontsize=8.5)
    ax.set_xlabel("False positive rate")
    ax.legend(title="AUC", fontsize=7, title_fontsize=7, loc="lower right")
axes[0].set_ylabel("True positive rate (sensitivity)")
fig.tight_layout()
S.save(fig, FIG / "f26_roc.png")

# protocol gap
fig, ax = plt.subplots(figsize=(6.4, 3.3))
names = list(MODELS)
yy = np.arange(len(names))[::-1]
for key, col, off in (("A", S.SAND, 0.22), ("B", S.SLATE, 0.0), ("C", S.OXBLOOD, -0.22)):
    v = [R["m7"][key]["models"][m]["auc"] for m in names]
    lo = [R["m7"][key]["models"][m]["auc"] - R["m7"][key]["models"][m]["auc_ci"][0] for m in names]
    hi = [R["m7"][key]["models"][m]["auc_ci"][1] - R["m7"][key]["models"][m]["auc"] for m in names]
    ax.errorbar(v, yy + off, xerr=[lo, hi], fmt="o", color=col, ecolor=col, capsize=2, markersize=5,
                label=f"Protocol {key}")
ax.set_yticks(yy, names)
ax.set_xlabel("Test ROC-AUC with 95% bootstrap interval")
ax.legend(loc="lower right")
ax.grid(axis="y", visible=False)
S.save(fig, FIG / "f27_protocol_gap.png")

# nearest-neighbour distance from test to train under both protocols
fig, ax = plt.subplots(figsize=(6.0, 2.5))
bins = np.logspace(-3, 1.2, 60)
for key, col in (("A", S.SAND), ("B", S.SLATE), ("C", S.OXBLOOD)):
    d_ = np.load(ROOT / "results" / f"nn_dist_{key}.npy")
    ax.hist(np.maximum(d_, 1e-3), bins=bins, histtype="stepfilled", alpha=0.5, color=col,
            label=f"Protocol {key}, median {np.median(d_):.2f}")
ax.set_xscale("log")
ax.set_xlabel("Distance from a test record to its closest training record (standardised units)")
ax.set_yticks([])
ax.legend()
S.save(fig, FIG / "f25_nn_distance.png")

# permutation importance of the best protocol-B model, and logistic coefficients
bestB = max(R["m7"]["C"]["models"], key=lambda m: R["m7"]["C"]["models"][m]["cv_auc_mean"])
pipeB = fitted[("C", bestB)][0]
PC = protocols["C"]
pi = permutation_importance(pipeB, np.r_[PC["Xtr"], PC["Xte"]], np.r_[PC["ytr"], PC["yte"]], scoring="roc_auc",
                            n_repeats=10, random_state=SEED, n_jobs=1)
feat_names = [LABEL[c] for c in NUM] + ["Male"]
pim = pd.DataFrame({"mean": pi.importances_mean, "sd": pi.importances_std}, index=feat_names).sort_values("mean")
tsave(pim.round(4), "perm_importance_C")
R["m7"]["perm_importance_C"] = {"model": bestB, "values": pim.round(4).to_dict(orient="index")}
lrB = fitted[("C", "Logistic regression")][0]
coefs = pd.Series(lrB.named_steps["clf"].coef_[0], index=feat_names)
R["m7"]["lr_coef_C"] = coefs.round(4).to_dict()
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
axes[0].barh(pim.index, pim["mean"], xerr=pim["sd"], color=S.OXBLOOD, height=0.6, ecolor=S.MUTED)
axes[0].set_title(f"Permutation importance, {SHORT[bestB]} (protocol C)")
axes[0].set_xlabel("Drop in test ROC-AUC when shuffled")
axes[0].grid(axis="y", visible=False)
cs = coefs.sort_values()
axes[1].barh(cs.index, cs.values, color=[S.OXBLOOD if v > 0 else S.TEAL for v in cs.values], height=0.6)
axes[1].axvline(0, color=S.MUTED, lw=0.8)
axes[1].set_title("Logistic regression coefficients (protocol C)")
axes[1].set_xlabel("Change in log-odds of disease per 1 SD")
axes[1].grid(axis="y", visible=False)
fig.tight_layout()
S.save(fig, FIG / "f28_importance.png")

# fairness check by gender for the best protocol-B model
fair = {}
for key_f in ("A", "C"):
    bm = max(R["m7"][key_f]["models"], key=lambda m: R["m7"][key_f]["models"][m]["cv_auc_mean"])
    pB, sB = fitted[(key_f, bm)][1], fitted[(key_f, bm)][2]
    sexB, yTe = protocols[key_f]["sex_te"], protocols[key_f]["yte"]
    fair[key_f] = {"model": bm}
    for sx in ("Male", "Female"):
        m = sexB == sx
        yt = yTe[m]
        tn, fp, fn, tp = confusion_matrix(yt, pB[m], labels=[0, 1]).ravel()
        fair[key_f][sx] = {"n": int(m.sum()), "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
                           "acc": float((tp + tn) / m.sum()), "auc": float(roc_auc_score(yt, sB[m]))}
R["m8_fairness"] = fair

R["env"] = {"python": __import__("sys").version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__, "xgboost": __import__("xgboost").__version__,
            "statsmodels": sm.__version__ if hasattr(sm, "__version__") else __import__("statsmodels").__version__,
            "seed": SEED, "runtime_s": round(time.time() - t0, 1)}
with open(ROOT / "results" / "results.json", "w") as f:
    json.dump(R, f, indent=1, default=float)
print(f"All done in {time.time() - t0:.0f}s")
