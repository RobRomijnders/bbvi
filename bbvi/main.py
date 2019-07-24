import numpy as np
from datetime import datetime
from os.path import join
import tensorflow as tf
from bbvi.util import DataLoader
from bbvi.model import MAPModel, VIReparamModel


def train_laplace():
    data_loader = DataLoader()
    model = MAPModel(num_features=data_loader.num_features)

    # Set up a session
    sess = tf.get_default_session()
    sess.run(model.init_op)

    for step in range(100000):
        X, y = data_loader.sample_batch()
        _, log_posterior_train, accuracy_train = \
            sess.run([model.train_step, model.log_p_theta_given_D, model.accuracy], {model.data: X, model.targets: y})

        if step % 500 == 0:
            X, y = data_loader.sample_batch(data_split='test')
            log_posterior_test, accuracy_test = \
                sess.run([model.log_p_theta_given_D, model.accuracy], {model.data: X, model.targets: y})

            print(f'At step {step:10.0f} '
                  f'Log posterior {log_posterior_train:8.3f}/{log_posterior_test:8.3f}  '
                  f'Accuracy {accuracy_train:5.3f}/{accuracy_test:5.3f}')

    # Using our trained model, let's find the posterior predictive
    # TODO Now do laplace integration here


def train_vi_reparam():
    data_loader = DataLoader()
    model = VIReparamModel(num_features=data_loader.num_features)

    # Set up a session
    sess = tf.get_default_session()
    sess.run(model.init_op)

    # Set up a writer for Tensorboard
    # Make writers to write to Tensorboard
    now = datetime.now()
    logdir = join('log_tb', now.strftime("%Y%m%d-%H%M%S"))
    writer = tf.summary.FileWriter(logdir=logdir)

    for step in range(50000):
        X, y = data_loader.sample_batch()
        _, log_posterior_train, accuracy_train, summary_str = \
            sess.run([model.train_step, model.log_p_theta_given_D, model.accuracy, model.summary_op], {model.data: X, model.targets: y})

        writer.add_summary(summary_str, step)
        writer.flush()

        if step % 100 == 0:
            X, y = data_loader.sample_batch(data_split='test')
            log_posterior_test, accuracy_test = \
                sess.run([model.log_p_theta_given_D, model.accuracy], {model.data: X, model.targets: y})

            print(f'At step {step:10.0f} '
                  f'Log posterior {log_posterior_train:8.3f}/{log_posterior_test:8.3f}  '
                  f'Accuracy {accuracy_train:5.3f}/{accuracy_test:5.3f} ')

    # Using our trained model, let's find the posterior predictive


if __name__ == '__main__':
    with tf.Session() as sess:
        # train_laplace()
        train_vi_reparam()
