from src.data_loader import load_data

def main():
    path = "data/df_minimization.csv"
    df = load_data(path)

    print(str(df))
    print("Local setup has been done")
    


if __name__ == "__main__":
    main()