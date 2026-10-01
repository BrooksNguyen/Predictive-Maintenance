"""
Preprocess NASA Turbofan (C-MAPSS) Dataset
───────────────────────────────────────────
Converts the space-delimited train_FD001.txt into a proper CSV with headers
that the IoT simulator can stream.

Columns (26 total, space-separated, no header in source):
  1  unit_number          → machine_id  (prefixed as "engine_XXX")
  2  time_in_cycles       → cycle
  3  operational_setting_1
  4  operational_setting_2
  5  operational_setting_3
  6–26  sensor_1 … sensor_21

Output:  iot_simulator/data/sensor_data.csv
"""

import csv
import os

DATASET_DIR = os.path.expanduser(
    "~/.cache/kagglehub/datasets/bishals098/"
    "nasa-turbofan-engine-degradation-simulation/versions/1"
)
SOURCE_FILE = os.path.join(DATASET_DIR, "train_FD001.txt")
OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "iot_simulator", "data", "sensor_data.csv"
)

COLUMNS = (
    ["machine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def main():
    rows_written = 0
    with open(SOURCE_FILE) as src, open(OUTPUT_FILE, "w", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerow(COLUMNS)

        for line in src:
            parts = line.strip().split()
            if len(parts) != 26:
                continue
            # Format unit_number as "engine_001"
            parts[0] = f"engine_{int(parts[0]):03d}"
            writer.writerow(parts)
            rows_written += 1

    print(f"✅  Wrote {rows_written} rows → {OUTPUT_FILE}")
    print(f"    Columns: {len(COLUMNS)}")


if __name__ == "__main__":
    main()
