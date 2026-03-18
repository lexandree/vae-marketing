import pandas as pd
import polars as pl
from pathlib import Path

def create_minimal_real_sample():
    data_dir = Path("data")
    output_dir = Path("data/minimal_sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load a small portion of transactions
    print("Loading transactions...")
    # Use pandas for initial sampling as it's often easier for head/random
    tx = pd.read_csv(data_dir / "transaction_data.csv", nrows=5000)
    
    # Keep only a few households for the minimal run
    target_households = tx['household_key'].unique()[:10]
    tx_sample = tx[tx['household_key'].isin(target_households)]
    
    # 2. Load products and filter to match transactions
    print("Loading products...")
    prod = pd.read_csv(data_dir / "product.csv")
    valid_prod_ids = tx_sample['PRODUCT_ID'].unique()
    prod_sample = prod[prod['PRODUCT_ID'].isin(valid_prod_ids)]
    
    # 3. Save samples
    tx_sample.to_csv(output_dir / "transactions_min.csv", index=False)
    prod_sample.to_csv(output_dir / "products_min.csv", index=False)
    
    print(f"Created minimal sample in {output_dir}")
    print(f"Transactions: {len(tx_sample)}, Products: {len(prod_sample)}")

if __name__ == "__main__":
    create_minimal_real_sample()
