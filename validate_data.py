import pandas as pd
import glob
import os

def report(df, name):
    print(f"\n=== {name} ===")
    print("Shape:", df.shape)
    print("Missing per column:\n", df.isna().sum())
    print("Total duplicates:", df.duplicated().sum())
    print("Data types:\n", df.dtypes)
    # Example sanity‐check: numeric columns should all be >= 0
    num_cols = df.select_dtypes(include="number").columns
    if not num_cols.empty:
        negs = (df[num_cols] < 0).sum()
        print("Negative values:\n", negs[negs>0])

def main():
    # Update these patterns to match your actual CSV/Parquet filenames:
    patterns = {
        "NASA battery (CSV)": "data/nasa_cleaned/*.csv",
        "EV population (Parquet)": "data/ev_population/*.parquet",
    }

    for label, pat in patterns.items():
        for path in glob.glob(pat):
            ext = os.path.splitext(path)[1].lower()
            if ext == ".csv":
                df = pd.read_csv(path)
            elif ext == ".parquet":
                df = pd.read_parquet(path)
            else:
                print(f"Skipping unrecognized file type: {path}")
                continue

            report(df, os.path.basename(path))


if __name__ == "__main__":
    main()
