"""Generate synthetic conveyor-belt shift datasets (normal and festival periods)."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def generate_dataset(n=350, festival=False, seed=42):
    rng = np.random.default_rng(seed)
    rated = rng.uniform(10, 25, n)
    belt_age = rng.uniform(0.2, 8, n)
    parcels_mean = 105 if not festival else 136.5  # +30% festival throughput
    parcels = rng.normal(parcels_mean, 18 if not festival else 22, n).clip(45, 190)
    vibration = rng.gamma(2.2, 1.8, n).clip(0.2, 12)
    dust = rng.beta(2, 3, n) * 100
    maintenance = rng.integers(0, 121, n)
    humidity = rng.uniform(25, 90, n)
    current = rated * (0.42 + parcels/260 + vibration/35 + belt_age/65
                       + rng.normal(0, 0.055, n))
    current = np.clip(current, 2, rated * 1.45)
    load = current / rated
    # Synthetic risk mechanism for a teaching dataset; not a real engineering limit.
    logit = (-5.0 + 2.15*load + 0.23*vibration + 0.018*dust
             + 0.020*maintenance + 0.24*belt_age + 0.012*(parcels-90)
             + 0.008*(humidity-50) + (0.45 if festival else 0))
    probability = 1 / (1 + np.exp(-logit))
    jam = rng.binomial(1, probability)
    frame = pd.DataFrame({
        "motor_current": current.round(3),
        "rated_motor_current": rated.round(3),
        "vibration_level": vibration.round(3),
        "belt_age": belt_age.round(2),
        "parcels_per_minute": parcels.round(1),
        "dust_accumulation_index": dust.round(1),
        "days_since_maintenance": maintenance,
        "humidity": humidity.round(1),
        "next_shift_jam_status": jam.astype(int)
    })
    # Small sensor missingness to demonstrate imputation.
    for column in ("vibration_level", "dust_accumulation_index", "humidity"):
        idx = rng.choice(n, size=max(1, int(n*0.025)), replace=False)
        frame.loc[idx, column] = np.nan
    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=350)
    args = parser.parse_args()
    out = ROOT / "datasets"
    out.mkdir(exist_ok=True)
    generate_dataset(args.rows, False, 42).to_csv(out/"conveyor_v1.csv", index=False)
    generate_dataset(args.rows, True, 84).to_csv(out/"conveyor_v2.csv", index=False)
    print(f"Created {args.rows} rows in each dataset version.")

if __name__ == "__main__":
    main()
