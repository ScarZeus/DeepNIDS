from src.data_loader import load_data
from src.preprocessing import preprocessing

def main():
    path = "data/df_minimization.csv"
    df = preprocessing(load_data(path))
    
    X = df.drop("Label", axis=1).values
    y = df["Label"].values

    X = X.reshape(X.shape[0],X.shape[1],1)
    print(X.shape)
    



if __name__ == "__main__":
    main()