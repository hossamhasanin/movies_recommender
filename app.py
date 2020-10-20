import os
from flask import *
import numpy as np
import pandas as pd
# import tensorflow as tf
# import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
from lite_encoder import LiteEncoder
tf.disable_v2_behavior()

app = Flask(__name__)

embeddings = np.load(
    "../data/movies_embed.npy", allow_pickle=True)
# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")

embed = LiteEncoder()
df = pd.read_csv("../data/movies.csv")
indcies = pd.Series(df.index, index=df["title"])
norm = np.linalg.norm(embeddings, axis=1)


@app.route('/', methods=['GET'])
def main():
    return "<h1>Welcome to our movie recommender api</h1>"


@app.route('/name', methods=['GET'])
def predic_based_on_name():
    # here we want to get the value of name (i.e. ?name=some-value)
    try:
        name = request.args.get('name')
        print(name)
        idx = indcies[name]
        sim = np.sum(embeddings[idx] * embeddings,
                     axis=1) / norm
        sim_scores = list(enumerate(sim))
        return predict_movies(sim_scores)
    except:
        return jsonify({"error": "Something went wrong", "state": 404})


@app.route('/desc', methods=['GET'])
def predic_based_on_desc():

    # # here we want to get the value of desc (i.e. ?desc=some-value)
    try:
        desc_movie = request.args.get('desc')
        # desc_embed = embed([desc_movie])
        desc_embed = embed.encode(
            sentences=[desc_movie])
        print(desc_embed)
        # sim = np.sum(desc_embed["outputs"] * embeddings,
        #              axis=1) / norm
        sim = np.sum(desc_embed * embeddings,
                     axis=1) / norm
        sim_scores = list(enumerate(sim))
        return predict_movies(sim_scores)
    except:
        return jsonify({"error": "Something went wrong", "state": 404})


def predict_movies(sim_scores):
    all_movies = df[['title', 'description', 'year', 'avg_vote']]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:10]
    movies = [i[0] for i in sim_scores]

    print(movies[0:10])

    recommend = all_movies.iloc[movies]
    results = []
    for _, row in recommend.iterrows():
        print(row)
        results.append(
            {"title": row["title"], "year": row["year"],
             "avg_vote": row["avg_vote"], "description": row["description"]})

    return jsonify({"results": results, "state": 200})
