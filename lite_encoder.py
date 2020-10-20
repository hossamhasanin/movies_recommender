import tensorflow.compat.v1 as tf
import sentencepiece as spm
import tensorflow_hub as hub
import numpy as np
from tqdm import tqdm


class LiteEncoder:
    def __init__(self,  model_link="https://tfhub.dev/google/universal-sentence-encoder-lite/2"):
        tf.disable_v2_behavior()
        self.graph = tf.Graph()
        with self.graph.as_default():
            module = hub.Module(model_link)
            self.input_placeholder = tf.sparse_placeholder(
                tf.int64, shape=[None, None])
            self.encodings = module(
                inputs=dict(
                    values=self.input_placeholder.values,
                    indices=self.input_placeholder.indices,
                    dense_shape=self.input_placeholder.dense_shape))

            with tf.Session() as sess:
                spm_path = sess.run(module(signature="spm_path"))

            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(spm_path)
            print("SentencePiece model loaded at {}.".format(spm_path))

    def process_to_IDs_in_sparse_format(self, sp, sentences):
        # An utility method that processes sentences with the sentence piece processor
        # 'sp' and returns the results in tf.SparseTensor-similar format:
        # (values, indices, dense_shape)
        ids = [sp.EncodeAsIds(x) for x in sentences]
        max_len = max(len(x) for x in ids)
        dense_shape = (len(ids), max_len)
        values = [item for sublist in ids for item in sublist]
        indices = [[row, col]
                   for row in range(len(ids)) for col in range(len(ids[row]))]
        return (values, indices, dense_shape)

    def encode(self, sentences):
        values, indices, dense_shape = self.process_to_IDs_in_sparse_format(
            self.sp, sentences)

        with tf.Session(graph=self.graph) as session:
            init = tf.initialize_all_variables()
            session.run(init)
            des_e = session.run(
                self.encodings,
                feed_dict={self.input_placeholder.values: values,
                           self.input_placeholder.indices: indices,
                           self.input_placeholder.dense_shape: dense_shape})
        return des_e

    def make_embeddings(self, data, file_name="embeddings", train=False):
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(),
                         tf.tables_initializer()])
            embeddings = []
            for x in tqdm(range(43)):
                f = x * 2000
                if x == 42:
                    t = len(data)
                else:
                    t = (x+1) * 2000
                values, indices, dense_shape = self.process_to_IDs_in_sparse_format(
                    self.sp, data[f:t])

                embeddings.append(session.run(
                    self.encodings,
                    feed_dict={self.input_placeholder.values: values,
                               self.input_placeholder.indices: indices,
                               self.input_placeholder.dense_shape: dense_shape}))
            embeddings = [item for sublist in embeddings for item in sublist]
        if train:
            np.save(file_name, embeddings)
        return embeddings
