from sklearn.model_selection import train_test_split
import pandas as pd

def train_val_split(train_val_csv, train_csv, val_csv, test_size=0.07, random_state=42):
    df = pd.read_csv(train_val_csv)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    return train_df, val_df