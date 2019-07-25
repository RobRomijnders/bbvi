from datetime import datetime
from os.path import join
from bbvi.util import DataLoader
from bbvi.model import MAPModel, VIModel

import numpy as np
import tensorflow as tf


def train_MAP():
    data_loader = DataLoader()
    model = MAPModel(num_features=data_loader.num_features)

    # Set up a session
    sess = tf.get_default_session()
    sess.run(model.init_op)

    for step in range(10000):
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

    # Make Laplace approximation for the covariance matrix.
    # See chapter 27 of Mackay's book or chapter 8.4 in Murphy's book
    hessian = sess.run(model.hessians, {model.data: X, model.targets: y})
    covariance_approx = -1 / np.squeeze(hessian[0])

    std_approx = np.sqrt(np.diag(covariance_approx))
    print('Laplace approximation for the posterior. Print std per parameter on next line')
    print(', '.join((f'{num:5.2f}' for num in std_approx)))


def train_vi():
    data_loader = DataLoader()
    model = VIModel(num_features=data_loader.num_features, gradient_method='score')

    # Set up a session
    sess = tf.get_default_session()
    sess.run(model.init_op)

    # Set up a writer for Tensorboard
    # Make writers to write to Tensorboard
    now = datetime.now()
    logdir = join('log_tb', now.strftime("%Y%m%d-%H%M%S"))
    writer = tf.summary.FileWriter(logdir=logdir)

    for step in range(20000):
        X, y = data_loader.sample_batch()
        _, log_posterior_train, accuracy_train, summary_str = \
            sess.run([model.train_step, model.log_p_theta_given_D, model.accuracy, model.summary_op], {
                model.data: X, model.targets: y})

        if step % 50 == 0:
            writer.add_summary(summary_str, step)
            writer.flush()

        if step % 500 == 0:
            X, y = data_loader.sample_batch(data_split='test')
            log_posterior_test, accuracy_test = \
                sess.run([model.log_p_theta_given_D, model.accuracy], {model.data: X, model.targets: y})

            print(f'At step {step:10.0f} '
                  f'Log posterior {log_posterior_train:10.1f}/{log_posterior_test:10.1f}  '
                  f'Accuracy {accuracy_train:5.3f}/{accuracy_test:5.3f} ')

    sigma_node = tf.get_collection("variational_sigma")[0]
    nu_sigma = sess.run(sigma_node, {model.data: X, model.targets: y})
    print('Variational approximation for the posterior. Print variational std per parameter on next line')
    print(', '.join((f'{num:5.2f}' for num in nu_sigma)))


if __name__ == '__main__':
    with tf.Session() as session:
        # train_MAP()
        train_vi()
