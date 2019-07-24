import tensorflow as tf
import numpy as np

from bbvi.util import get_random_normal_variable

"""
Here we define three inference setup's for the Binary Bayesian Logistic regression.
There's is a lot of repeated code, but I think this copy-pasting makes it at least clear what is happening in each
inference setup
"""


class MAPModel:
    def __init__(self, num_features):
        self.data = tf.placeholder(tf.float32, [None, num_features], "data")
        self.targets = tf.placeholder(tf.float32, [None, ], 'targets')

        # Extend the dataset with a column of ones. The parameter for this column will serve as the bias
        data_ext = tf.concat((self.data, tf.ones((tf.shape(self.data)[0], 1))), axis=1)

        # Do the multiplication
        self.W = tf.get_variable('W', [num_features + 1, 1])
        activation = tf.matmul(data_ext, self.W)

        # Use built in cross entropy from logits. It uses some tricks to prevent overflow near the boundaries
        log_likelihood = - tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=activation, labels=tf.expand_dims(self.targets, axis=-1)))

        # Take into account the prior
        variance_prior = 10**2
        log_prior = (num_features + 1) / 2 + tf.log(2 * np.pi * variance_prior) - 1 / (2 * variance_prior) * tf.reduce_sum(tf.pow(self.W, 2))

        # Get the posterior probability for our parameter
        self.log_p_theta_given_D = log_likelihood + log_prior

        # Set up a training schedule
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(0.05, global_step, 5000, 0.1)

        # This is a list with only self.W, but let's keep consistent with other implementations ;)
        tvars = tf.trainable_variables()
        gradients = tf.gradients(-self.log_p_theta_given_D, tvars)

        self.train_step = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(gradients, tvars))
        self.init_op = tf.global_variables_initializer()

        # Calculate accuracy for interpretation
        self.prediction = tf.squeeze(tf.cast(activation > 0, tf.float32), axis=1)
        approx_equal = tf.cast(tf.abs(self.prediction - self.targets) < 1E-4, tf.float32)
        self.accuracy = tf.reduce_mean(approx_equal)


class VIReparamModel:
    def __init__(self, num_features):
        self.data = tf.placeholder(tf.float32, [None, num_features], "data")
        self.targets = tf.placeholder(tf.float32, [None, ], 'targets')

        # Extend the dataset with a column of ones. The parameter for this column will serve as the bias
        data_ext = tf.concat((self.data, tf.ones((tf.shape(self.data)[0], 1))), axis=1)

        num_mc = 13

        # Define the variational parameters nu_*
        W, nu_mu, sigma, nu_pre_sigma = get_random_normal_variable('weight', [num_features + 1, ], num_samples=num_mc)

        # Note:
        # We take MC samples over W. So all tensors related to W will be broadcast with a dimension of length [num_mc, ]

        # Log probability of the parameters under the variational approximation
        log_q = tf.reduce_sum(- 1 / 2 * tf.log(2 * np.pi) - tf.log(sigma)
                              - 1 / (2 * sigma ** 2) * tf.pow(W - nu_mu, 2), axis=-1)  # Tensor in [num_mc, ]

        # Do the multiplication
        activation = tf.matmul(data_ext, tf.transpose(W))  # Tensor in [batch_size, num_mc, ]

        # Use built in cross entropy from logits. It uses some tricks to prevent overflow near the boundaries
        labels_repeat = tf.tile(tf.expand_dims(self.targets, axis=-1), [1, num_mc])
        self.log_likelihood = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=activation, labels=labels_repeat), axis=0)  # Tensor in [num_mc, ]

        # Take into account the prior
        variance_prior = 10**1
        self.log_prior = (num_features + 1) / 2 + tf.log(2 * np.pi * variance_prior) - 1 / (2 * variance_prior) * tf.reduce_sum(tf.pow(W, 2), axis=-1)  # Tensor in [num_mc, ]

        # Get the posterior probability for our parameter
        self.log_p_theta_given_D = tf.reduce_mean(self.log_likelihood + self.log_prior)  # TODO portion the log_prior over all batches?

        # Set up a training schedule
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(0.0005, global_step, 500, 0.1)

        # Write out the gradient
        d_log_q_d_mu = tf.pow(sigma, -2) * (W - nu_mu)
        d_log_q_d_pre_sigma = tf.pow(sigma, -1) * (tf.pow(sigma, -2)*tf.pow(W-nu_mu, 2) - 1) * tf.sigmoid(nu_pre_sigma)

        # Credits to David Blei for this name :p
        elbo_instant = self.log_likelihood + self.log_prior - log_q
        elbo_instant = tf.expand_dims(elbo_instant, axis=-1)

        # Keep consistent with other implementations and define tvars and gradients as lists
        # Intuition for these gradients: Discriminate cases where `elbo_instant is positive or negative
        #  - If probability under the model is higher than the probability under q, then follow the gradient of q
        #  - If probability under the model is lower than the probability under q, then walk against the gradient of q
        tvars = [nu_mu, nu_pre_sigma]
        gradient_mu = d_log_q_d_mu * elbo_instant  # tensor in [num_mc, num_param]
        gradient_pre_sigm = d_log_q_d_pre_sigma * elbo_instant  # tensor in [num_mc, num_param]

        gradients = [tf.reduce_mean(gradient_mu, axis=0), tf.reduce_mean(gradient_pre_sigm, axis=0)]

        # Negate the gradients, because tf implementation follows negative gradient
        # https://github.com/tensorflow/tensorflow/issues/27499
        gradients = [-1 * grad for grad in gradients]

        # Now follow the tensorflow technicalities
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate).apply_gradients(zip(gradients, tvars))
        self.init_op = tf.global_variables_initializer()

        # Calculate accuracy for interpretation
        self.prediction = tf.cast(tf.reduce_mean(activation, axis=1) > 0, tf.float32)
        approx_equal = tf.cast(tf.abs(self.prediction - self.targets) < 1E-4, tf.float32)
        self.accuracy = tf.reduce_mean(approx_equal)

        # Summaries for Tensorboard
        tf.summary.histogram("Weights", W)
        tf.summary.histogram("Variational_mean", nu_mu)
        tf.summary.histogram("Variational_sigma", sigma)

        tf.summary.scalar("Log_likelihood", tf.reduce_mean(self.log_likelihood))
        tf.summary.scalar("Log_prior", tf.reduce_mean(self.log_prior))
        tf.summary.scalar("Accuracy", self.accuracy)
        tf.summary.scalar("ELBO_inst", tf.reduce_mean(elbo_instant))

        # Find std of gradient estimates. For clarity, we do not summarize statistics for each parameter, that would be
        # messy. Therefore, we average over all variational parameters, \phi
        tf.summary.scalar("Grad_ave_var_mu", tf.reduce_mean(tf.math.reduce_std(gradient_mu, axis=0)), family="grad_ave_var")
        tf.summary.scalar("Grad_ave_var_psigma", tf.reduce_mean(tf.math.reduce_std(gradient_pre_sigm, axis=0)), family="grad_ave_var")

        self.summary_op = tf.summary.merge_all()
