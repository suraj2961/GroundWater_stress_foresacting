"""Generate result visualizations from the processed forecasting outputs.

Run from the project root:

    .venv\Scripts\python.exe results_visualization\visualize_results.py

Charts are written to results_visualization\outputs so they remain separate
from the pipeline outputs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results_visualization" / "outputs"

DISTRICTS = [
    "Bengaluru Urban",
    "Mysuru",
    "Ballari",
    "Uttara Kannada",
    "Kalaburagi",
    "Vijayapura",
]


def plot_forecasts() -> None:
    historical = pd.read_csv(DATA_DIR / "panel_dataset.csv", parse_dates=["date"])
    forecast = pd.read_csv(
        DATA_DIR / "multi_district_forecast_24month.csv", parse_dates=["date"]
    )

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=False)
    for axis, district in zip(axes.flat, DISTRICTS):
        history = historical[
            (historical["district"] == district)
            & (historical["date"] >= "2021-01-01")
        ].sort_values("date")
        prediction = forecast[forecast["district"] == district].sort_values("date")

        axis.plot(
            history["date"],
            history["GRACE_TWS_anomaly_cm"],
            color="#34495e",
            linewidth=1.5,
            label="Historical",
        )
        axis.plot(
            prediction["date"],
            prediction["predicted_GRACE_TWS_anomaly_cm"],
            color="#3498db",
            linewidth=1.5,
            linestyle="--",
            label="Forecast",
        )
        axis.axvspan(
            prediction["date"].min(),
            prediction["date"].max(),
            alpha=0.08,
            color="#3498db",
        )
        axis.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        axis.set_title(district, fontsize=11, fontweight="bold")
        axis.set_ylabel("GRACE TWS Anomaly (cm)", fontsize=9)
        axis.tick_params(axis="x", rotation=45, labelsize=8)
        axis.legend(fontsize=8, loc="upper right")
        axis.grid(alpha=0.3)

    fig.suptitle(
        "Groundwater Storage: Historical and 24-Month Forecast",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "forecast_by_district.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stress_categories() -> None:
    forecast = pd.read_csv(DATA_DIR / "multi_district_forecast_24month.csv")
    category_order = ["Safe", "Semi-Critical", "Critical", "Over-Exploited"]
    category_colors = {
        "Safe": "#2ecc71",
        "Semi-Critical": "#f1c40f",
        "Critical": "#e67e22",
        "Over-Exploited": "#e74c3c",
    }
    summary = (
        forecast.groupby(["district", "stress_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=DISTRICTS, columns=category_order, fill_value=0)
    )

    fig, axis = plt.subplots(figsize=(12, 6))
    bottom = pd.Series(0, index=summary.index, dtype="int64")
    for category in category_order:
        axis.bar(
            summary.index,
            summary[category],
            bottom=bottom,
            label=category,
            color=category_colors[category],
        )
        bottom += summary[category]

    axis.set_ylabel("Forecasted Months (out of 24)")
    axis.set_title("Forecasted Groundwater Stress Category Distribution", fontweight="bold")
    axis.legend(title="Stress Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stress_category_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_by_horizon() -> None:
    accuracy = pd.read_csv(DATA_DIR / "horizon_accuracy_summary.csv")

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(accuracy["horizon_months"], accuracy["mean_MAE"], marker="o", label="Model MAE")
    axis.plot(
        accuracy["horizon_months"],
        accuracy["mean_naive_MAE"],
        marker="o",
        label="Naive MAE",
    )
    axis.set_xlabel("Forecast Horizon (months)")
    axis.set_ylabel("Mean Absolute Error")
    axis.set_title("Forecast Accuracy by Horizon", fontweight="bold")
    axis.set_xticks(accuracy["horizon_months"])
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_by_horizon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance() -> None:
    importance = pd.read_csv(DATA_DIR / "feature_importance.csv")
    importance = importance.iloc[:, :2].copy()
    importance.columns = ["feature", "importance"]
    importance = importance.sort_values("importance").tail(15)

    fig, axis = plt.subplots(figsize=(10, 7))
    axis.barh(importance["feature"], importance["importance"], color="#3498db")
    axis.set_xlabel("Importance")
    axis.set_title("Top 15 Feature Importance Scores", fontweight="bold")
    axis.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_forecasts()
    plot_stress_categories()
    plot_accuracy_by_horizon()
    plot_feature_importance()
    print(f"Saved result visualizations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
