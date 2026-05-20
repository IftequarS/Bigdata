##Bigdata and Cloud Computing. Demonstrating of serverless Cloud Computing
***************CODES FOR DATASET CLEANING AND VISULIZATION OF THE DATASETS****************

from urllib.parse import quote
from azureml.core import Workspace, Dataset, Datastore
import pandas as pd
import os
 
base_url = "https://bdccstroage.blob.core.windows.net/bdcccontainer/"
file_name = "raw_ibm_cloud.csv"
 
ws = Workspace.from_config()
 
url = base_url + quote(file_name)
 
df = pd.read_csv(url)
 
print(df.head())
print(df.shape)
df = df.iloc[:, 0].str.split(",", expand=True)
 
df.columns = [
    "interval_start",
    "location",
    "kind",
    "host",
    "method",
    "statuscode",
    "endpoint",
    "aggregated_stats_name",
    "aggregated_stats_value"
]
 
df["interval_start"] = pd.to_datetime(
    df["interval_start"],
    unit="s",
    errors="coerce"
)
 
df["statuscode"] = pd.to_numeric(
    df["statuscode"],
    errors="coerce"
)
 
df["aggregated_stats_value"] = pd.to_numeric(
    df["aggregated_stats_value"],
    errors="coerce"
)
 
df = df.drop_duplicates()
 
df.fillna({
    "location": "unknown",
    "kind": "unknown",
    "host": "unknown",
    "method": "unknown",
    "endpoint": "unknown",
}, inplace=True)
 
df["statuscode"].fillna(
    df["statuscode"].median(),
    inplace=True
)
 
df["aggregated_stats_value"].fillna(
    df["aggregated_stats_value"].median(),
    inplace=True
)
 
clean_path = "./outputs/ibm_cloud_cleaned.csv"
 
os.makedirs("outputs", exist_ok=True)
 
df.to_csv(clean_path, index=False)
 
datastore = ws.get_default_datastore()
 
datastore.upload(
    src_dir="outputs",
    target_path="raw-data",
    overwrite=True
)
 
dataset = Dataset.Tabular.from_delimited_files(
    path=(datastore, "outputs/ibm_cloud_cleaned.csv"),
    validate=False,
    infer_column_types=False,
    separator=",",
    header=True
)
 
dataset = dataset.register(
    workspace=ws,
    name="ibm_cloud_cleaned",
    description="Cleaned metrics dataset with schema inference disabled",
    create_new_version=True
)
 
print("Dataset registered successfully")
 
#CODE FOR DATASET 
 
import matplotlib.pyplot as plt
 
import matplotlib.ticker as mticker
 
import numpy as np
 
import warnings
 
warnings.filterwarnings("ignore")
 
plt.rcParams.update({
 
    "figure.dpi": 130,
 
    "axes.titlesize": 14,
 
    "axes.labelsize": 12,
 
    "xtick.labelsize": 10,
 
    "ytick.labelsize": 10,
 
    "legend.fontsize": 10,
 
    "axes.grid": True,
 
    "grid.alpha": 0.3,
 
})
 
TEAL   = "#0A9396"
 
NAVY   = "#0D1B3E"
 
GOLD   = "#E9C46A"
 
RED    = "#E63946"
 
GREEN  = "#2A9D8F"
 
COLORS = [TEAL, NAVY, GOLD, RED, GREEN, "#A8DADC", "#457B9D"]
 
# ── Load Data ─────────────────────────────────────────────────────────────────
 
df = pd.read_csv("cleaned_ibm_cloud.csv")
 
df["interval_start"] = pd.to_datetime(df["interval_start"], errors="coerce")
 
df_avg = df[df["aggregated_stats_name"] == "avg"].copy()
 
df_avg["is_error"] = df_avg["statuscode"].apply(
 
    lambda x: "Error (4xx/5xx)" if x >= 400 or x == -1 else "Success (2xx/3xx)"
 
)
 
print(f"Rows loaded: {len(df):,}")
 
#  FIGURE 1 — HTTP Status Code Distribution
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
fig.suptitle("Figure 1: HTTP Status Code Distribution", fontsize=15, fontweight="bold")
 
# 1a: Bar chart of status codes
 
status_counts = df["statuscode"].value_counts().sort_index()
 
axes[0].bar(status_counts.index.astype(str), status_counts.values, color=TEAL, edgecolor="white")
 
axes[0].set_title("1a — Count per Status Code")
 
axes[0].set_xlabel("HTTP Status Code")
 
axes[0].set_ylabel("Number of Records")
 
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
 
axes[0].tick_params(axis="x", rotation=45)
 
# 1b: Success vs Error pie
 
success_err = df_avg["is_error"].value_counts()
 
axes[1].pie(
 
    success_err.values,
 
    labels=success_err.index,
 
    autopct="%1.1f%%",
 
    colors=[GREEN, RED],
 
    startangle=140,
 
    wedgeprops={"edgecolor": "white", "linewidth": 2},
 
)
 
axes[1].set_title("1b — Success vs Error Rate")
 
plt.tight_layout()
 
plt.savefig("fig1_status_distribution.png", bbox_inches="tight")
 
plt.show()
 
print("INSIGHT 1:")
 
print(f"  Status 200 count : {status_counts.get(200, 0):,}")
 
print(f"  Error rate       : {(success_err.get('Error (4xx/5xx)', 0) / success_err.sum() * 100):.1f}%")
 
#  FIGURE 2 — Request Volume by HTTP Method
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
fig.suptitle("Figure 2: Request Volume by HTTP Method", fontsize=15, fontweight="bold")
 
method_counts = df["method"].value_counts()
 
axes[0].bar(method_counts.index, method_counts.values,
 
            color=COLORS[:len(method_counts)], edgecolor="white")
 
axes[0].set_title("2a — Total Requests per Method")
 
axes[0].set_xlabel("HTTP Method")
 
axes[0].set_ylabel("Number of Records")
 
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
 
# 2b: Stacked bar — method × kind
 
method_kind = df.groupby(["method", "kind"]).size().unstack(fill_value=0)
 
bottom = np.zeros(len(method_kind))
 
for i, col in enumerate(method_kind.columns):
 
    axes[1].bar(method_kind.index, method_kind[col], bottom=bottom,
 
                label=col, color=COLORS[i], edgecolor="white")
 
    bottom += method_kind[col].values
 
axes[1].set_title("2b — Requests by Method & Kind")
 
axes[1].set_xlabel("HTTP Method")
 
axes[1].set_ylabel("Number of Records")
 
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
 
axes[1].legend(title="Kind")
 
plt.tight_layout()
 
plt.savefig("fig2_method_distribution.png", bbox_inches="tight")
 
plt.show()
 
print("INSIGHT 2:")
 
for m, c in method_counts.items():
 
    print(f"  {m}: {c:,} requests ({c/len(df)*100:.1f}%)")
 
#  FIGURE 3 — Average Latency by Location
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
fig.suptitle("Figure 3: Latency by Datacenter Location", fontsize=15, fontweight="bold")
 
loc_latency = (df_avg.groupby("location")["aggregated_stats_value"]
 
               .mean().sort_values(ascending=False))
 
axes[0].bar(loc_latency.index, loc_latency.values,
 
            color=COLORS[:len(loc_latency)], edgecolor="white")
 
axes[0].set_title("3a — Mean Latency per Location (µs)")
 
axes[0].set_xlabel("Location")
 
axes[0].set_ylabel("Mean Latency (µs)")
 
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
 
axes[0].tick_params(axis="x", rotation=30)
 
# 3b: Boxplot using matplotlib
 
clip_99 = df_avg["aggregated_stats_value"].quantile(0.99)
 
loc_order = (df_avg.groupby("location")["aggregated_stats_value"]
 
             .median().sort_values(ascending=False).index)
 
data_for_box = [
 
    df_avg[df_avg["location"] == loc]["aggregated_stats_value"]
 
    .clip(upper=clip_99).values
 
    for loc in loc_order
 
]
 
bp = axes[1].boxplot(data_for_box, patch_artist=True, labels=loc_order)
 
for patch, color in zip(bp["boxes"], COLORS):
 
    patch.set_facecolor(color)
 
    patch.set_alpha(0.7)
 
axes[1].set_title("3b — Latency Distribution per Location (99th pct clip)")
 
axes[1].set_xlabel("Location")
 
axes[1].set_ylabel("Latency (µs)")
 
axes[1].tick_params(axis="x", rotation=30)
 
plt.tight_layout()
 
plt.savefig("fig3_latency_by_location.png", bbox_inches="tight")
 
plt.show()
 
print("INSIGHT 3:")
 
print(f"  Highest latency: {loc_latency.idxmax()} ({loc_latency.max():,.0f} µs)")
 
print(f"  Lowest  latency: {loc_latency.idxmin()} ({loc_latency.min():,.0f} µs)")
 
#  FIGURE 4 — Latency by HTTP Method
 
fig, ax = plt.subplots(figsize=(10, 5))
 
fig.suptitle("Figure 4: Average Latency by HTTP Method", fontsize=15, fontweight="bold")
 
method_latency = (df_avg.groupby("method")["aggregated_stats_value"]
 
                  .mean().sort_values(ascending=False))
 
bars = ax.bar(method_latency.index, method_latency.values,
 
              color=COLORS[:len(method_latency)], edgecolor="white")
 
ax.set_xlabel("HTTP Method")
 
ax.set_ylabel("Mean Latency (µs)")
 
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
 
for bar in bars:
 
    ax.text(bar.get_x() + bar.get_width() / 2,
 
            bar.get_height() + 500,
 
            f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=9)
 
plt.tight_layout()
 
plt.savefig("fig4_latency_by_method.png", bbox_inches="tight")
 
plt.show()
 
 
 
#  FIGURE 5 — Latency Over Time
 
 
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
 
fig.suptitle("Figure 5: Latency Trends Over Time", fontsize=15, fontweight="bold")
 
daily = (df_avg.set_index("interval_start")
 
         .resample("D")["aggregated_stats_value"].mean().dropna())
 
axes[0].plot(daily.index, daily.values, color=TEAL, linewidth=1.5)
 
axes[0].fill_between(daily.index, daily.values, alpha=0.15, color=TEAL)
 
axes[0].set_title("5a — Daily Mean Latency (µs)")
 
axes[0].set_ylabel("Latency (µs)")
 
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
 
for i, (loc, grp) in enumerate(df_avg.groupby("location")):
 
    daily_loc = (grp.set_index("interval_start")
 
                 .resample("D")["aggregated_stats_value"].mean())
 
    axes[1].plot(daily_loc.index, daily_loc.values,
 
                 label=loc, linewidth=1.2, color=COLORS[i % len(COLORS)])
 
axes[1].set_title("5b — Daily Mean Latency by Location")
 
axes[1].set_ylabel("Latency (µs)")
 
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
 
axes[1].legend(loc="upper right", ncol=2)
 
plt.tight_layout()
 
plt.savefig("fig5_latency_over_time.png", bbox_inches="tight")
 
plt.show()
 
print("INSIGHT 5:")
 
print(f"  Peak day : {daily.idxmax().date()} ({daily.max():,.0f} µs)")
 
print(f"  Lowest day: {daily.idxmin().date()} ({daily.min():,.0f} µs)")
 
 
 
#  FIGURE 6 — Top 10 Slowest Endpoints
 
 
fig, ax = plt.subplots(figsize=(12, 6))
 
fig.suptitle("Figure 6: Top 10 Slowest Endpoints", fontsize=15, fontweight="bold")
 
top_ep = (df_avg.groupby("endpoint")["aggregated_stats_value"]
 
          .mean().sort_values(ascending=False).head(10).sort_values())
 
ax.barh(top_ep.index, top_ep.values, color=TEAL, edgecolor="white")
 
ax.set_xlabel("Mean Latency (µs)")
 
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
 
ax.set_ylabel("Endpoint")
 
plt.tight_layout()
 
plt.savefig("fig6_slowest_endpoints.png", bbox_inches="tight")
 
plt.show()
 
 
#  FIGURE 7 — Error Rate by Location & Method
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
fig.suptitle("Figure 7: Error Rate Analysis", fontsize=15, fontweight="bold")
 
err_by_loc = (df_avg.groupby("location")["is_error"]
 
              .apply(lambda x: (x == "Error (4xx/5xx)").sum() / len(x) * 100)
 
              .sort_values(ascending=False))
 
axes[0].bar(err_by_loc.index, err_by_loc.values, color=RED, edgecolor="white")
 
axes[0].set_title("7a — Error Rate (%) by Location")
 
axes[0].set_xlabel("Location")
 
axes[0].set_ylabel("Error Rate (%)")
 
axes[0].tick_params(axis="x", rotation=30)
 
err_by_method = (df_avg.groupby("method")["is_error"]
 
                 .apply(lambda x: (x == "Error (4xx/5xx)").sum() / len(x) * 100)
 
                 .sort_values(ascending=False))
 
axes[1].bar(err_by_method.index, err_by_method.values,
 
            color=COLORS[:len(err_by_method)], edgecolor="white")
 
axes[1].set_title("7b — Error Rate (%) by HTTP Method")
 
axes[1].set_xlabel("HTTP Method")
 
axes[1].set_ylabel("Error Rate (%)")
 
plt.tight_layout()
 
plt.savefig("fig7_error_rates.png", bbox_inches="tight")
 
plt.show()
 
print("INSIGHT 7:")
 
print(f"  Highest error location : {err_by_loc.idxmax()} ({err_by_loc.max():.1f}%)")
 
print(f"  Highest error method   : {err_by_method.idxmax()} ({err_by_method.max():.1f}%)")
 
#  FIGURE 8 — Heatmap: Latency by Location × Method (matplotlib only)
 
fig, ax = plt.subplots(figsize=(10, 5))
 
fig.suptitle("Figure 8: Heatmap — Mean Latency by Location × Method (µs)",
 
             fontsize=14, fontweight="bold")
 
pivot = (df_avg.groupby(["location", "method"])["aggregated_stats_value"]
 
         .mean().unstack(fill_value=0))
 
im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
 
ax.set_xticks(range(len(pivot.columns)))
 
ax.set_xticklabels(pivot.columns)
 
ax.set_yticks(range(len(pivot.index)))
 
ax.set_yticklabels(pivot.index)
 
plt.colorbar(im, ax=ax, label="Mean Latency (µs)")
 
for i in range(len(pivot.index)):
 
    for j in range(len(pivot.columns)):
 
        ax.text(j, i, f"{pivot.values[i, j]:,.0f}",
 
                ha="center", va="center", fontsize=8,
 
                color="black" if pivot.values[i, j] < pivot.values.max() * 0.7 else "white")
 
ax.set_xlabel("HTTP Method")
 
ax.set_ylabel("Location")
 
plt.tight_layout()
 
plt.savefig("fig8_heatmap.png", bbox_inches="tight")
 
plt.show()
 
#  FIGURE 9 — Aggregated Stats Type Distribution
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
fig.suptitle("Figure 9: Aggregated Stats Distribution", fontsize=15, fontweight="bold")
 
stats_count = df["aggregated_stats_name"].value_counts()
 
axes[0].bar(stats_count.index, stats_count.values,
 
            color=COLORS[:len(stats_count)], edgecolor="white")
 
axes[0].set_title("9a — Records per Stat Type")
 
axes[0].set_xlabel("Stat Name")
 
axes[0].set_ylabel("Count")
 
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
 
axes[0].tick_params(axis="x", rotation=30)
 
clip_99 = df["aggregated_stats_value"].quantile(0.99)
 
stat_names = df["aggregated_stats_name"].unique()
 
data_box = [df[df["aggregated_stats_name"] == s]["aggregated_stats_value"]
 
            .clip(upper=clip_99).values for s in stat_names]
 
bp = axes[1].boxplot(data_box, patch_artist=True, labels=stat_names)
 
for patch, color in zip(bp["boxes"], COLORS):
 
    patch.set_facecolor(color)
 
    patch.set_alpha(0.7)
 
axes[1].set_title("9b — Value Distribution per Stat Type (99th pct clip)")
 
axes[1].set_xlabel("Stat Name")
 
axes[1].set_ylabel("Value")
 
axes[1].tick_params(axis="x", rotation=30)
 
plt.tight_layout()
 
plt.savefig("fig9_stats_distribution.png", bbox_inches="tight")
 
plt.show()
 
#  FIGURE 10 — CLIENT vs SERVER Latency
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
fig.suptitle("Figure 10: CLIENT vs SERVER Latency Comparison", fontsize=15, fontweight="bold")
 
kind_latency = df_avg.groupby("kind")["aggregated_stats_value"].mean()
 
axes[0].bar(kind_latency.index, kind_latency.values,
 
            color=[TEAL, NAVY], edgecolor="white")
 
axes[0].set_title("10a — Mean Latency: CLIENT vs SERVER")
 
axes[0].set_xlabel("Kind")
 
axes[0].set_ylabel("Mean Latency (µs)")
 
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
 
for i, (kind, grp) in enumerate(df_avg.groupby("kind")):
 
    daily_k = (grp.set_index("interval_start")
 
               .resample("D")["aggregated_stats_value"].mean())
 
    axes[1].plot(daily_k.index, daily_k.values,
 
                 label=kind, linewidth=1.5, color=COLORS[i])
 
axes[1].set_title("10b — Daily Latency: CLIENT vs SERVER")
 
axes[1].set_ylabel("Mean Latency (µs)")
 
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
 
axes[1].legend()
 
plt.tight_layout()
 
plt.savefig("fig10_client_vs_server.png", bbox_inches="tight")
 
plt.show()
 
print("INSIGHT 10:")
 
for k, v in kind_latency.items():
 
    print(f"  {k}: {v:,.0f} µs mean latency")
 
print("\n All 10 figures generated successfully.")
