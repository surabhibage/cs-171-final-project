import pandas as pd

url = "hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv"
df = pd.read_csv(url)
df.to_csv("data/spotify-tracks-dataset.csv", index=False, header=False)