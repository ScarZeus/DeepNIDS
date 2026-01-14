import pandas as pd


def preprocessing(df):
    execption_col = 'Label'
    
    for col in df.columns:
        if col != execption_col:
            df[col] = df[col].astype("float32")
    df[execption_col] = df[execption_col].astype("int8")

    DROP_COLS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Timestamp",
    "Fwd Header Length.1"
    ]
    
    df.drop(columns= DROP_COLS, inplace=True, errors= "ignore")
    return df