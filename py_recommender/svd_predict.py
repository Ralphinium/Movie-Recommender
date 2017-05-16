import tensorflow as tf
import numpy as np

def predict(user, item):
    with tf.variable_scope("biases", reuse=True):
        user_bias_tensor = tf.get_default_graph().get_tensor_by_name("biases/user_bias_tensor:0")
        item_bias_tensor = tf.get_default_graph().get_tensor_by_name("biases/item_bias_tensor:0")

    with tf.variable_scope("latent_factors", reuse=True):
        user_latent_tensor = tf.get_default_graph().get_tensor_by_name("latent_factors/user_latent_tensor:0")
        item_latent_tensor = tf.get_default_graph().get_tensor_by_name("latent_factors/item_latent_tensor:0")

    with tf.variable_scope("mean", reuse=True):
        global_mean = tf.get_default_graph().get_tensor_by_name("mean/global_mean:0")

    bu = tf.nn.embedding_lookup(user_bias_tensor, user)
    bi = tf.nn.embedding_lookup(item_bias_tensor, item)

    pu = tf.nn.embedding_lookup(user_latent_tensor, user, name="pu")
    qi = tf.nn.embedding_lookup(item_latent_tensor, item, name="qi")

    return global_mean + bu + bi + tf.reduce_sum(qi * pu)


def read_ratings_matrix(filename):
    df = np.loadtxt(filename, delimiter=",")
    return df


def cos_sim(x, y):
    """
    Returns the cosine similarity of two numpy arrays.
    For this similarity measure, we consider NaN values to be zero.
    :param x: 
    :param y: 
    :return: 
    """
    dot = np.dot(x, y)
    ssx = np.sum(np.power(x, 2))
    ssy = np.sum(np.power(y, 2))
    return dot / np.sqrt(ssx * ssy)


def cos_sim_df(df, x):
    """
    Returns the index of the array in the given dataframe that is most similar to the given array.
    :param df: 
    :param x: 
    :return: 
    """
    max_sim = 0
    max_id = -1
    for i in range(len(df)):
        curr_sim = cos_sim(df[i], x)
        if max_sim < curr_sim:
            max_sim = curr_sim
            max_id = i
        if i % 1000 == 0:
            print("yo")
    return max_sim, max_id

if __name__ == "__main__":
    ratings = read_ratings_matrix("./data/user_ratings_matrix.csv")
    test = np.zeros(3952)
    test[0] = 0.853154
    test[47] = 0.853154
    test[149] = 0.853154

    _, sim_id = cos_sim_df(ratings, test)
    print(sim_id)

    ratings = []
    print("Predicting")
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./SVD_model/model-90000.meta')
    ckpt = tf.train.get_checkpoint_state('./SVD_model/')
    saver.restore(sess, tf.train.latest_checkpoint('./SVD_model'))
    for i in range(0, 3952):
        ratings.append(sess.run(predict(sim_id, i)))

