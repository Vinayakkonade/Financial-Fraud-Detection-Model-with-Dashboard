import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df.fillna(0, inplace=True)
    df['amount'] = np.log1p(df['amount'])
    df = pd.get_dummies(df, columns=['transaction_type', 'merchant_category'])
    scaler = StandardScaler()
    df[['amount', 'transaction_count']] = scaler.fit_transform(df[['amount', 'transaction_count']])
    return df, scaler