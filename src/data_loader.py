# src/data_loader.py
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo
from src.config import DATA_DIR, CSV_FILENAME

def load_data() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    target_csv = DATA_DIR / CSV_FILENAME

    if target_csv.exists() and os.path.getsize(target_csv) > 100_000:
        print(f"Loading local cached dataset from: {target_csv}")
        df = pd.read_csv(target_csv)
    else:
        print("Fetching dataset directly via official UCI client...")
        try:
            # Bank Marketing dataset
            bank_marketing = fetch_ucirepo(id=222)
            X = bank_marketing.data.features
            y = bank_marketing.data.targets
            df = pd.concat([X, y], axis=1)
            
            # Save cache
            df.to_csv(target_csv, index=False)
            print(f"Saved dataset locally to: {target_csv}")
        except Exception as e:
            raise RuntimeError(f"UCI Repo fetch failed: {e}") from e

    # Clean column names (strip spaces, lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalize the source dataset's numeric day-of-month field. Some CSV mirrors
    # label this field `day_of_week`, while the UCI source labels it `day`.
    # The app and model use the stable internal name `day_of_week` for compatibility.
    if 'day' in df.columns and 'day_of_week' not in df.columns:
        df = df.rename(columns={'day': 'day_of_week'})

    # Target column check and binarize
    target_col = 'y' if 'y' in df.columns else df.columns[-1]
    df['y'] = (df[target_col].astype(str).str.strip().str.lower().isin(['yes', '1', 'true'])).astype(int)
    if target_col != 'y':
        df = df.drop(columns=[target_col])

    # Avoid data leakage
    df = df.drop(columns=['duration'], errors='ignore')

    return df

if __name__ == "__main__":
    data = load_data()
    print("Dataset Ready! Shape:", data.shape)
    print("Available Features:", [c for c in data.columns if c != 'y'])