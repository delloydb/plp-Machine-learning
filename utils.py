"""
utils.py
Helper functions for encoding and scaling in the crop yield prediction project.
"""

import pandas as pd

def encode_soil_type(df):
    """
    Encodes the 'soil_type' column using one-hot encoding.
    Drops the first category to avoid multicollinearity.
    """
    if 'soil_type' in df.columns:
        df_encoded = pd.get_dummies(df, columns=['soil_type'], drop_first=True)
        return df_encoded
    else:
        raise KeyError("Column 'soil_type' not found in the DataFrame.")
