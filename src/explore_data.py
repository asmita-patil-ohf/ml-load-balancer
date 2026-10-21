import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "data" / "load_balancer_dataset.csv"
PLOT_DIR = ROOT_DIR / "data" / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

# 1. Response-time distribution
plt.figure(figsize=(8, 5))
plt.hist(df["response_time_ms"], bins=40)
plt.xlabel("Response time (ms)")
plt.ylabel("Number of requests")
plt.title("Distribution of Response Times")
plt.tight_layout()
plt.savefig(PLOT_DIR / "response_time_distribution.png")
plt.show()

# 2. Average latency per server
server_latency = (
    df.groupby("server_id")["response_time_ms"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(8, 5))
server_latency.plot(kind="bar")
plt.xlabel("Server")
plt.ylabel("Average response time (ms)")
plt.title("Average Latency by Server")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(PLOT_DIR / "server_latency.png")
plt.show()

# 3. CPU utilization versus response time
plt.figure(figsize=(8, 5))
plt.scatter(
    df["cpu_percent"],
    df["response_time_ms"],
    alpha=0.5,
)
plt.xlabel("CPU utilization (%)")
plt.ylabel("Response time (ms)")
plt.title("CPU Utilization vs Response Time")
plt.tight_layout()
plt.savefig(PLOT_DIR / "cpu_vs_latency.png")
plt.show()

# 4. Correlations between numeric features
numeric_df = df.select_dtypes(include="number")
correlations = numeric_df.corr()

print("\nCorrelation with response time:")
print(
    correlations["response_time_ms"]
    .sort_values(ascending=False)
    .round(3)
)

plt.figure(figsize=(9, 7))
plt.imshow(correlations, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(
    range(len(correlations.columns)),
    correlations.columns,
    rotation=75,
    ha="right",
)
plt.yticks(range(len(correlations.columns)), correlations.columns)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(PLOT_DIR / "correlation_matrix.png")
plt.show()

print(f"\nPlots saved to: {PLOT_DIR}")