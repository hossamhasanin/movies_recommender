import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
from nltk.tokenize import RegexpTokenizer
import numpy as np


df = pd.read_csv("movies.csv")


def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(str(text))
    text = " ".join(text)
    return text


df["description"] = df["description"].apply(func=remove_punctuation)

desc = df["description"].tolist()

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")
embeddings = embed(desc)

dev = input("test or train ? > ")
if dev == "train":
    np.save("movies_embed", embeddings["outputs"])

mode = input("Search by story or movie names ? > ")

sim_scores = None
if mode == "description":
    desc_movie = input("Movie desc > ")
    desc_embed = embed([desc_movie])
    sim = np.sum(desc_embed["outputs"] * embeddings["outputs"],
                 axis=1) / np.linalg.norm(embeddings["outputs"], axis=1)
    sim_scores = list(enumerate(sim))
elif mode == "names":
    indcies = pd.Series(df.index, index=df["title"])
    title = input("Movie name > ")
    idx = indcies[title]
    sim = np.sum(embeddings["outputs"][idx] * embeddings["outputs"],
                 axis=1) / np.linalg.norm(embeddings["outputs"], axis=1)
    sim_scores = list(enumerate(sim))

all_movies = df[['title', 'description', 'year']]
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
sim_scores = sim_scores[0:10]
movies = [i[0] for i in sim_scores]

print(movies[0:10])

recommend = all_movies.iloc[movies]

for index, row in recommend.iterrows():
    print("title > " + row["title"])
    print("year > " + str(row["year"]))
    print("description > " + row["description"]+"\n")
