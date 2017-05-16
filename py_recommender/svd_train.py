import time
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2

import svd_ops
import data_input

# np.random.seed(13575)

# Constants.
BATCH_SIZE = 1000
USER_NUM = 6040
ITEM_NUM = 3952
DIM = 20
EPOCH_MAX = 100
DEVICE = "/gpu:0"


def clip(x):
    """
    Clips all the values in x to be between 1.0 and 5.0
    :param np.array x: The array of values to be clipped. 
    :return np.array: An array the same size as `x` with clipped values.
    """
    return np.clip(x, 1.0, 5.0)


def get_data(filename, sep="::", ratio=0.9):
    """
    Reads the data from the filename, and returns a training and testing dataframe.
    :param str filename: Name of the file containing the data. 
    :param str sep: The separator in the file.
    :param int ratio: The ratio between the size of the training and testing dataframes.
    :return: The training and testing dataframes.
    """
    df = data_input.read_data(filename, sep=sep)
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * ratio)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    print("Finished reading data.")
    return df_train, df_test


def svd(train, test):
    per_batch = len(train) // BATCH_SIZE

    # Generators of batches for the training and testing sets.
    train_gen = data_input.Batcher([train.user, train.item, train.rate], batch_size=BATCH_SIZE)
    test_gen = data_input.Batcher([test.user, test.item, test.rate], batch_size=len(test))

    user_batch = tf.placeholder(tf.int32, shape=[None], name="user_ids")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="item_ids")
    rate_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = svd_ops.predict_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM,
                                             dim=DIM, device=DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = svd_ops.optimize_svd(infer, regularizer, rate_batch, learning_rate=0.002, reg=0.05, device=DEVICE)

    init_op = tf.global_variables_initializer()

    print("Starting training.")
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="/SVD", graph=sess.graph)

        saver = tf.train.Saver(tf.trainable_variables())

        print("{} {} {} {}".format("Epoch", "Train Error", "Test Error", "Elapsed Time"))
        errors = deque(maxlen=per_batch)
        start = time.time()

        for i in range(EPOCH_MAX * per_batch):
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
    df_train, df_test = get_data(filename="./data/ml-latest-100k/ratings.csv", sep=",")
    svd(df_train, df_test)
    print("Done!")
