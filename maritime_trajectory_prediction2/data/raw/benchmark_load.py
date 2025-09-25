#!/usr/bin/env python3
import argparse
import time


def benchmark_pandas(file_path, engine):
    import pandas as pd

    start = time.perf_counter()
    _ = pd.read_parquet(file_path, engine=engine)
    duration = time.perf_counter() - start
    print(f"Pandas ({engine}) load: {duration:.3f} s")


def benchmark_polars(file_path):
    try:
        import polars as pl
    except ImportError:
        print("Polars not installed; skipping Polars benchmark.")
        return
    start = time.perf_counter()
    _ = pl.read_parquet(file_path)
    duration = time.perf_counter() - start
    print(f"Polars load:         {duration:.3f} s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Parquet load times")
    parser.add_argument(
        "--file", "-f", required=True, help="Path to your .parquet file"
    )
    args = parser.parse_args()

    print(f"Benchmarking Parquet loads for {args.file!r}\n")

    # Pandas with pyarrow
    try:
        benchmark_pandas(args.file, engine="pyarrow")
    except Exception as e:
        print(f"Pandas (pyarrow) load failed: {e}")

    # Pandas with fastparquet
    try:
        benchmark_pandas(args.file, engine="fastparquet")
    except Exception as e:
        print(f"Pandas (fastparquet) load failed: {e}")

    # Polars (if available)
    benchmark_polars(args.file)


if __name__ == "__main__":
    main()
