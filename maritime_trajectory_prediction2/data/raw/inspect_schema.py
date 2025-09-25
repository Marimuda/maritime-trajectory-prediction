#!/usr/bin/env python3
import polars as pl

# 1) Read in the entire Parquet (zero‐copy, parallel)
df = pl.read_parquet("ais_data.parquet")

# 2) Print the column names
print("Columns:")
for c in df.columns:
    print("  ", c)

# 3) Print the schema (column → dtype)
print("\nSchema:")
print(df.schema)
