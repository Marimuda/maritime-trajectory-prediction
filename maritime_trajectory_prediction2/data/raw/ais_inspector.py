#!/usr/bin/env python3
import argparse
import json
import time

import pandas as pd
from tqdm import tqdm


def extract_ais_messages(log_path, sample=None):
    """
    Generator: yields dicts for each AIS JSON message found.
    Skips lines that don't contain a JSON payload starting with {"class":"AIS"}.
    """
    with open(log_path, errors="ignore") as f:
        for idx, line in enumerate(f):
            if sample and idx >= sample:
                break
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            ts, rest = parts
            if '{"class":"AIS"' not in rest:
                continue
            json_start = rest.find("{")
            if json_start < 0:
                continue
            payload = rest[json_start:].strip().rstrip(",")
            try:
                msg = json.loads(payload)
            except json.JSONDecodeError:
                continue
            msg["timestamp"] = ts
            yield msg


def main():
    parser = argparse.ArgumentParser(
        description="Extract AIS JSON records and benchmark load times"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to .log file")
    parser.add_argument(
        "--output", "-o", required=True, help="Output Parquet file path"
    )
    parser.add_argument(
        "--sample",
        "-n",
        type=int,
        default=None,
        help="Process only this many lines (for testing)",
    )
    args = parser.parse_args()

    # Estimate total lines for progress bar
    total_lines = None
    if args.sample is None:
        try:
            with open(args.input, "rb") as f:
                total_lines = sum(1 for _ in f)
        except Exception:
            total_lines = None

    # Extraction
    records = []
    iterator = extract_ais_messages(args.input, sample=args.sample)
    if total_lines:
        iterator = tqdm(iterator, total=total_lines, desc="Extracting messages")
    else:
        iterator = tqdm(iterator, desc="Extracting messages")
    for msg in iterator:
        records.append(msg)

    # DataFrame creation
    df = pd.DataFrame.from_records(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # Convert object columns to pandas StringDtype for compatibility
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("string")
    # Convert numeric-like columns
    for col in df.columns:
        if col != "timestamp" and pd.api.types.is_string_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # Benchmark sections
    start = time.perf_counter()
    df.to_parquet(args.output, compression="snappy", index=False)
    write_duration = time.perf_counter() - start
    start = time.perf_counter()
    _ = pd.read_parquet(args.output)
    read_duration = time.perf_counter() - start

    print(f"Parquet write: {write_duration:.2f}s")
    print(f"Parquet read:  {read_duration:.2f}s")


if __name__ == "__main__":
    main()
