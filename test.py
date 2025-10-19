aime2024_path="/u/haoboxu/work/verl/data/aime2024/test.parquet"
aime2025_path="/u/haoboxu/work/verl/data/aime2025/test.parquet"
amc23_path="/u/haoboxu/work/verl/data/amc23/test.parquet"
dapo17k_path="/u/haoboxu/work/verl/data/amc23/test.parquet"
olympiadbench_path="/u/haoboxu/work/verl/data/olympiadbench/test.parquet"
omnimath_path="/u/haoboxu/work/verl/data/omnimath/test.parquet"
math500_path="/u/haoboxu/work/verl/data/math500/test.parquet"
minerva_path="/u/haoboxu/work/verl/data/minervamath/test.parquet"
gsm8k_path="/u/haoboxu/work/verl/data/gsm8k/test.parquet"


VAL_FILES=[aime2024_path, aime2025_path, amc23_path, gsm8k_path, olympiadbench_path, math500_path, minerva_path]

# read
import pandas as pd
def read_parquet(file_path):
    df = pd.read_parquet(file_path)
    return df

for file in VAL_FILES:
    df = read_parquet(file)
    print(f"Dataset: {file}, Number of samples: {len(df)}")