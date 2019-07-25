import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal


class DataLoader:
    """
    Small wrapper to abstract all code relating to loading data
    """
    def __init__(self, batch_size=16):
        # Wine data set
        dataset = load_wine()

        # For now, just create a binary classification problem
        selection = dataset.target < 2

        X, y = dataset.data[selection], dataset.target[selection]
        y = self.random_flip(y, 0.1)

        # Dummy data set
        # Uncomment these lines to create a dummy data sets of two highly separable point clouds
        # num_feat = 8
        # num_half = 500
        # X1 = multivariate_normal(5 * np.ones((8,)), np.eye(num_feat)).rvs(num_half)
        # X2 = multivariate_normal(-5 * np.ones((8,)), np.eye(num_feat)).rvs(num_half)
        #
        # X = np.concatenate((X1, X2), axis=0)
        # y = np.concatenate((np.zeros((num_half)), np.ones((num_half))), axis=0)

        self.data = dict()
        self.data['X_train'], self.data['X_test'], self.data['y_train'], self.data['y_test'] = train_test_split(X, y)

        self.mean, self.std = None, None
        # self._normalize_data()

        self.batch_size = batch_size

    @property
    def num_features(self):
        return self.data['X_train'].shape[1]

    @staticmethod
    def random_flip(data, portion):
        """
        Randomly flip a portion of the binary labels. To spice up the problem a bit :)
        :param data:
        :param portion:
        :return:
        """
        # Establish the sizes
        num_samples = len(data)
        num_flip = int(num_samples * portion)

        # Select random indices to flip
        idx = np.random.choice(num_samples, num_flip, replace=False)

        # Do the flipping
        data[idx] = (data[idx] - 1/2) * -1 + 1/2
        return data

    def _normalize_data(self):
        # Calculate the first and second moment from the train data
        self.mean = np.mean(self.data['X_train'], axis=0)
        self.std = np.std(self.data['X_train'], axis=0)

        # Standardize the training data
        self.data['X_train'] -= self.mean
        self.data['X_train'] /= self.std

        # Standardize the test data
        self.data['X_test'] -= self.mean
        self.data['X_test'] /= self.std

    def sample_batch(self, data_split='train'):
        # Sample from batch
        datasplit_size = len(self.data['y_' + data_split])
        idx = np.random.choice(datasplit_size, self.batch_size, replace=False)

        return self.data['X_' + data_split][idx], self.data['y_' + data_split][idx]


def get_random_normal_variable(name, shape, dtype=tf.float32, num_samples=13):
    """
    Create weight tensors with factorized Gaussian approximation of each element.

    Define the standard deviation behind a softplus to enforce positivity


    Credits for code inspiration: https://github.com/DeNeutoy/bayesian-rnn/
    :param name: Name for the corresponding tf variables
    :param shape: shape for the variable. Note that weights are sampled and thus have +1 dimension
    :param dtype: dtype for the variables involved
    :param num_samples: number of samples from the variational distro over W
    :return:
    """

    # Inverse of a softplus function, so that the value of the standard deviation
    # will be equal to what the user specifies, but we can still enforce positivity
    # by wrapping the standard deviation in the softplus function.
    # standard_dev = tf.log(tf.exp(standard_dev) - 1.0) * tf.ones(shape)

    # it's important to initialize variances with care, otherwise the model takes too long to converge
    sigma_min = 1-1/10
    sigma_max = 1+1/10

    rho_max_init = tf.log(tf.exp(sigma_max) - 1.0)
    rho_min_init = tf.log(tf.exp(sigma_min) - 1.0)
    std_init = tf.random_uniform_initializer(rho_min_init, rho_max_init)

    # Initialize the mean
    mean = tf.get_variable(name + "_mean", shape, dtype=dtype)

    # Initialize the standard deviation
    pre_sigma = tf.get_variable(name + "_standard_deviation",
                                         shape,
                                         initializer=std_init,
                                         dtype=dtype)

    standard_deviation = tf.nn.softplus(pre_sigma) + 1e-5

    # The famous reparametrization formula for the factorized Gaussian
    noise = tf.random_normal([num_samples] + shape, 0.0, 1.0, dtype)
    weights = mean + standard_deviation * noise

    return weights, mean, standard_deviation, pre_sigma, noise