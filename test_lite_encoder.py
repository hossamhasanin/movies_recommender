import pandas as pd
from nltk.tokenize import RegexpTokenizer
import numpy as np
from lite_encoder import LiteEncoder
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

df = pd.read_csv("../data/movies.csv")

input_placeholder = tf.sparse_placeholder(
    tf.int64, shape=[None, None])


def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(str(text))
    text = " ".join(text)
    return text


df["description"] = df["description"].apply(func=remove_punctuation)

desc = df["description"].tolist()

embed = LiteEncoder(input_placeholder=input_placeholder)

dev = input("test or train ? > ")
if dev == "train":
    embeddings = embed.make_embeddings(
        data=desc, file_name="movies_embed", train=True)
else:
    embeddings = np.load(
        "../data/movies_embed.npy", allow_pickle=True)

mode = input("Search by story or movie names ? > ")

sim_scores = None
if mode == "description":
    desc_movie = input("Movie desc > ")
    desc_embed = embed.encode(
        sentences=[desc_movie], input_placeholder=input_placeholder)
    sim = np.sum(desc_embed * embeddings,
                 axis=1) / np.linalg.norm(embeddings, axis=1)
    sim_scores = list(enumerate(sim))

elif mode == "names":
    indcies = pd.Series(df.index, index=df["title"])
    title = input("Movie name > ")
    idx = indcies[title]
    sim = np.sum(embeddings[idx] * embeddings,
                 axis=1) / np.linalg.norm(embeddings, axis=1)
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
