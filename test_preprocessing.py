from utils.preprocessing import load_and_preprocess

X = load_and_preprocess('data/CICIDS2017.csv')
print(f"Data shape: {X.shape}")
print(f"Sample row: {X[0]}")
