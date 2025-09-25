#!/usr/bin/env python3
"""
Inspect Polars DataFrame schema and report column types,
identifying categorical columns and their categories (if low cardinality).
"""

import argparse

import polars as pl


def main():
    parser = argparse.ArgumentParser(description="Inspect parquet columns")
    parser.add_argument("-f", "--file", required=True, help="Path to Parquet file")
    parser.add_argument(
        "-c",
        "--cat_thresh",
        type=int,
        default=1000,
        help="Max unique values to treat as categorical",
    )
    args = parser.parse_args()

    # Load schema only
    df = pl.read_parquet(args.file)
    schema = df.schema
    print("Column Summary for:", args.file, end="\n\n")

    for name, dtype in schema.items():
        print(f"Column: {name}")
        dtype_str = str(dtype)
        print(f"  Type: {dtype_str}")
        # Check numeric vs non-numeric
        if dtype_str in [
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
            "Float32",
            "Float64",
        ]:
            print("  Category: Numeric")
        else:
            # Treat Utf8 and other as potential categorical
            unique_count = df[name].n_unique()
            print(f"  Unique values: {unique_count}")
            if unique_count <= args.cat_thresh:
                print("  Treated as categorical. Categories:")
                # Collect and print categories
                cats = df[name].unique().sort()
                for v in cats:
                    print(f"    - {v}")
            else:
                print("  Treated as non-categorical (high cardinality)")
        print()


if __name__ == "__main__":
    main()
