"""
The :mod:`training` module implements the training of SVD models on user-item-rating triples, using `Simon Funk's
<http://sifter.org/~simon/journal/20061211.html>`_ *SVD* algorithm. 
"""
import time
from collections import deque

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.core.framework import summary_pb2


def read_data(filename, sep="\t"):
    """
    Reads the ratings from the rating file, given by the filename.
    :param filename: The name of the rating file.
    :param sep: The seperator of the file.
    :return: A Pandas dataframe containing all the ratings.
    """
    col_names = ['user', 'item', 'rate', 'ts']
    df = pd.read_csv(filename, sep=sep, header=0, names=col_names)
    df['user'] -= 1
    df['item'] -= 1
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df['rate'] = df['rate'].astype(np.float32)
    return df


class Batcher:
    """
    Given a dataframe, this Batcher creates an iterable of random batches of the data.
    """
    def __init__(self, inputs, batch_size=10):
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        self.inputs = np.transpose([np.array(self.inputs[i]) for i in range(self.num_cols)])

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size, ))
        out = self.inputs[ids, :]
        return [out[:, i] for i in range(self.num_cols)]


def predict_svd(user_batch, item_batch, user_num, item_num, dim=5, device="/cpu:0"):
    """
    Let :math:`\\hat{r}_{ui}` be the predicted rating of user :math:`u` to item :math:`i` and finally let:

    .. math::
        e_{ui} = r_{ui} - \hat{r}_{ui}.

    This algorithm predicts :math:`\\hat{r}_{ui}` by:

    .. math::
        \hat{r}_{ui} = \mu + b_u + b_i + q_i^Tp_u

    where
    * :math:`b_u, b_i` are biases for user :math:`u` and item :math:`i` respectively,
    * :math:`q_i, p_u` are vectors of latent factors for user :math:`u` and item :math:`i` respectively.

    :param int user_batch: The user ids of the users.
    :param int item_batch: The item ids of the items.
    :param int user_num: How many users in the whole dataset.
    :param int item_num: How many items in the whole dataset.
    :param int dim: Number of latent factors to compute with.
    :param str device: Which device to use for the computations.
    """
    # Find global variables first.
    with tf.device("/cpu:0"):
        # Tensors for individual user and item biases.
        with tf.variable_scope("biases"):
            user_bias_tensor = tf.get_variable("user_bias_tensor", shape=[user_num])
            item_bias_tensor = tf.get_variable("item_bias_tensor", shape=[item_num])

        # Tensors for the latent factors of the users and items.
        with tf.variable_scope("latent_factors"):
            user_latent_tensor = tf.get_variable("user_latent_tensor", shape=[user_num, dim],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            item_latent_tensor = tf.get_variable("item_latent_tensor", shape=[item_num, dim],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.02))

        # The global mean of the data set.
        with tf.variable_scope("mean"):
            global_mean = tf.get_variable("global_mean", shape=[])

        # The biases of the users and items in this batch.
        bu = tf.nn.embedding_lookup(user_bias_tensor, user_batch, name="bu")
        bi = tf.nn.embedding_lookup(item_bias_tensor, item_batch, name="bi")

        # The latent factors of the users and items in this batch.
        pu = tf.nn.embedding_lookup(user_latent_tensor, user_batch, name="pu")
        qi = tf.nn.embedding_lookup(item_latent_tensor, item_batch, name="qi")

    # Calculate a prediction.
    with tf.device(device):
        predicted = bu + bi + tf.reduce_sum(pu * qi, 1)
        predicted = tf.add(predicted, global_mean, name="predicted_value")
        bias_loss = tf.nn.l2_loss(bu) + tf.nn.l2_loss(bi)
        latent_loss = tf.nn.l2_loss(pu) + tf.nn.l2_loss(qi)
        regularizer = tf.add(bias_loss, latent_loss, name="regularizer")

    return predicted, regularizer


def optimize_svd(predicted, regularizer, actual_batch, learning_rate=0.001, reg=0.1, device="/cpu:0"):
    """
    Trains using the ``train_set`` by minimizing the following error:

    .. math::
        \sum_{r_{ui} \in \R_{train}} \left(r_{ui} - \hat{r}_{ui} \\right)^2 +
        \lambda\\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2\\right)
    :param predicted: A tensor containing the predicted ratings.
    :param regularizer: A tensor containing the part of the error.
    :param actual_batch: The ratings of the users to the items.
    :param learning_rate: Learning rate.
    :param reg: Regularizer.
    :param device: Which device to use for the computations.
    :return: 
    """
    global_step = tf.train.get_global_step()
    assert global_step is not None

    with tf.device(device):
        reg = tf.constant(reg, dtype=tf.float32, name="l2_reg")
        cost = tf.nn.l2_loss(predicted - actual_batch)
        cost = tf.add(cost, tf.multiply(regularizer, reg), name="cost")
        train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(cost, global_step=global_step)

    return cost, train_op


def get_data(filename, sep="::", ratio=0.9):
    """
    Reads the data from the filename, and returns a training and testing dataframe.
    :param str filename: Name of the file containing the data. 
    :param str sep: The separator in the file.
    :param int ratio: The ratio between the size of the training and testing dataframes.
    :return: The training and testing dataframes.
    """
    df = read_data(filename, sep=sep)
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * ratio)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    print("Finished reading data.")
    return df_train, df_test


def clip(x):
    """
    Clips all the values in x to be between 1.0 and 5.0
    :param np.array x: The array of values to be clipped. 
    :return np.array: An array the same size as `x` with clipped values.
    """
    return np.clip(x, 1.0, 5.0)


def train(filename, batch_size, user_num, item_num, dim, epochs, device="/gpu:0"):
    df_train, df_test = get_data(filename, sep=',')
    per_batch = len(df_train) // batch_size

    # Generators of batches for the training and testing sets.
    train_gen = Batcher([df_train.user, df_train.item, df_train.rate], batch_size=batch_size)
    test_gen = Batcher([df_test.user, df_test.item, df_test.rate], batch_size=len(df_test))

    # Placeholders for the batches.
    user_batch = tf.placeholder(tf.int32, shape=[None], name="user_ids")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="item_ids")
    rate_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = predict_svd(user_batch, item_batch, user_num=user_num, item_num=item_num,
                                     dim=dim, device=device)
    _ = tf.contrib.framework.get_or_create_global_step()
    _, train_op = optimize_svd(infer, regularizer, rate_batch, learning_rate=0.002, reg=0.1, device=device)

    init_op = tf.global_variables_initializer()

    print("Starting training.")
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="./SVD_model", graph=sess.graph)

        saver = tf.train.Saver(tf.trainable_variables())

        print("{} {} {} {}".format("Epoch", "Train Error", "Test Error", "Elapsed Time"))
        errors = deque(maxlen=per_batch)
        start = time.time()

        for i in range(epochs * per_batch):
            users, items, ratings = next(train_gen)
            _, preds = sess.run([train_op, infer],
                                feed_dict={user_batch: users, item_batch: items, rate_batch: ratings})
            errors.append(np.power(preds - ratings, 2))

            if i % per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_errs = np.array([])

                users, items, ratings = next(test_gen)
                preds = sess.run(infer, feed_dict={user_batch: users, item_batch: items})

                preds = clip(preds)
                test_errs = np.append(test_errs, np.power(preds - ratings, 2))

                end = time.time()
                test_err = np.sqrt(np.mean(test_errs))
                print("{:3d} {:f} {:f} {:f}(s)".format(i // per_batch, train_err, test_err,
                                                       end - start))
                train_err_summary = summary_pb2.Summary(value=[
                    summary_pb2.Summary.Value(tag="training_error", simple_value=train_err)
                ])
                test_err_summary = summary_pb2.Summary(value=[
                    summary_pb2.Summary.Value(tag="test_error", simple_value=test_err)
                ])

                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end
        saver.save(sess, "./SVD_model/model", global_step=tf.train.get_global_step())


if __name__ == "__main__":
    # Constants.
    BATCH_SIZE = 1000
    USER_NUM = 671
    ITEM_NUM = 9123
    DIM = 15
    EPOCH_MAX = 100
    DEVICE = "/gpu:0"

    train("./data/ml-latest-100k/ratings.csv", batch_size=BATCH_SIZE, user_num=USER_NUM, item_num=ITEM_NUM,
          dim=DIM, epochs=EPOCH_MAX, device=DEVICE)
    print("Done.")
