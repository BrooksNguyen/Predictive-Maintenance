"""
IoT Sensor Data Simulator
──────────────────────────
Reads sensor_data.csv and streams each row over a TCP socket on port 9999.
When the CSV is exhausted it loops back to the start, simulating an
infinite real-time sensor feed.

Protocol:  one CSV line per TCP message, newline-delimited.
Format:    machine_id,timestamp,sensor_1,sensor_2,sensor_3,sensor_4
"""

import csv
import itertools
import logging
import os
import socket
import time

# ── Configuration ───────────────────────────────────────────
HOST = "0.0.0.0"
PORT = int(os.getenv("SIMULATOR_PORT", 9999))
DATA_PATH = os.getenv("DATA_PATH", "/app/data/sensor_data.csv")
SEND_INTERVAL = float(os.getenv("SEND_INTERVAL", 1.0))  # seconds between rows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [SIMULATOR]  %(message)s",
)
log = logging.getLogger(__name__)


def load_csv_rows(path: str) -> list[str]:
    """Load the CSV and return a list of raw data lines (no header)."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header
        return [",".join(row) for row in reader]


def serve(host: str, port: int, rows: list[str]) -> None:
    """Accept TCP connections and stream CSV rows in a loop."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    log.info("Listening on %s:%d — waiting for Spark to connect …", host, port)

    while True:
        conn, addr = server.accept()
        log.info("Connection from %s", addr)
        try:
            for row in itertools.cycle(rows):
                message = row + "\n"
                conn.sendall(message.encode("utf-8"))
                log.info("→  %s", row)
                time.sleep(SEND_INTERVAL)
        except (BrokenPipeError, ConnectionResetError):
            log.warning("Client disconnected — waiting for reconnection …")
        finally:
            conn.close()


if __name__ == "__main__":
    csv_rows = load_csv_rows(DATA_PATH)
    log.info("Loaded %d rows from %s", len(csv_rows), DATA_PATH)
    serve(HOST, PORT, csv_rows)
