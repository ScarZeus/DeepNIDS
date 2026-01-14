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
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Fwd Avg Bulk Rate"
    ]
    
    df.drop(columns= DROP_COLS, inplace=True, errors= "ignore")
    return df