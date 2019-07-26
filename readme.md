# Gradient estimation

Long time I struggled to understand how a gradient could have noisy estimates. To me, a gradient felt like a constant. Follow the chain rule, and you find a matrix with derivatives. However, rummaging in the world of Variational Inference, we arrive at situations where we have to estimate gradients. There exist no analytical expression. This project concerns these situations.

Variational inference turns an inference problem into an optimization problem. For the inference, we are interested in a posterior distribution. This distribution is too complex for our inference. We will make a simpler approximation to it. Finding the best approximation to the complex posterior poses us then with an optimization problem: optimize the approximation to be like the true, complex, posterior.

We rely on gradients to perform this optimization. Our optimization toolbox has many more approaches, but gradients suit us just fine. Gradients are favoured by researchers endeavouring on __black box variational inference (BBVI)__. BBVI aims to provide inference to people who write down arbitrary models. We require then only the model is differentiable. Fortunately, many popular models are differentiable. Especially with auto differentiation packages, finding these gradients are easy. 

## So why estimate the gradient?

Our objective for Variational Inference reads:

<img alt="$\mathcal{L}(\phi) = KL(q(z;\phi)||p(z|x)) = E_{q(z;\phi)}[\log \frac{q(z;\phi)}{p(z|x)}]$" src="https://github.com/robromijnders/bbvi/blob/master/svgs/6d5711c7397a215de3ae45da6c05be98.svg" align="middle" width="335.819055pt" height="33.20559pt"/>.

Now we want to optimize over the variational parameters, <img alt="$\phi$" src="https://github.com/robromijnders/bbvi/blob/master/svgs/f50853d41be7d55874e952eb0d80c53e.svg" align="middle" width="9.794565000000006pt" height="22.831379999999992pt"/>, such that this divergence is minimized. One foremost choice would be gradient descent. Now for _gradient_ descent, we need our gradient. 

We have two candidate method to estimate this gradient. Recently, [this paper](https://arxiv.org/abs/1906.10652) nicely outlined the differences. 

These two methods motivated me to write this project. I heard many people claim that for the Gaussian case, the pathwise derivative has less variance than the score function gradient. This main question we will tackle here.

### Score function gradient
The first method feels very like reinforcement learning. The gradient reads like:

<img alt="$\nabla_\phi \mathcal{L}(\phi) = E_{q(z;\phi)}[(\log p(z|x) - \log q(z;\phi)) \nabla_\phi \log q(z;\phi)]$" src="https://github.com/robromijnders/bbvi/blob/master/svgs/e970fdc2b4c7142da05d459d42901687.svg" align="middle" width="401.145855pt" height="24.65759999999998pt"/>

Intuitively, one can read this gradient as follows: first take samples from the approximate posterior. If the true posterior is higher than the approximation, follow the gradient. If the true posterior is lower than the approximation, walk against the gradient. 

This intuition feels to me very much like reinforcement learning. As soon as decision have been made, then update the model if the update was positive or negative. This wide deviation also explains why this estimator is known to have much higher variance from the second method.

The derivation follows from pushing the gradient in the expectation and write out the product rule for gradients. You can read it in full in the appendix of [this paper](https://arxiv.org/pdf/1401.0118.pdf).

### Pathwise derivative
The second method relies on the reparametrization trick. We appreciate that our variational approximation is really a shift and scale from the standard Gaussian distribution. Therefore, we can directly _backpropagate_ into these shift and scale parameters, rather than updating them in an RL like setting. 

The gradient thus reads like:
<img alt="$\nabla_\phi \mathcal{L}(\phi) = E_{s(\epsilon)}[\nabla_z(\log p(z|x) - \log q(z;\phi)) \nabla_\phi t(\epsilon, \phi)]$" src="https://github.com/robromijnders/bbvi/blob/master/svgs/324242f7b0e39ada3a5a6d1e4073391b.svg" align="middle" width="378.887355pt" height="24.65759999999998pt"/>

Note that here we take expectation over a base distribution, <img alt="$s(\epsilon)$" src="https://github.com/robromijnders/bbvi/blob/master/svgs/af8653192af20922eafa84d4dd90157c.svg" align="middle" width="27.16329pt" height="24.65759999999998pt"/>. We changed this via the reparametrization formula: 

<img alt="$z \sim q(z;\phi) \iff z = t(\epsilon, \phi), \epsilon \sim s(\epsilon)$" src="https://github.com/robromijnders/bbvi/blob/master/svgs/53e9d59ace20ff314129e59eab4ceca9.svg" align="middle" width="260.70775499999996pt" height="24.65759999999998pt"/>. 

For the Gaussian case, the reparametrization formula is:

<img alt="$t(\epsilon, \phi) = \mu_\phi + \sigma_\phi \epsilon$" src="https://github.com/robromijnders/bbvi/blob/master/svgs/068132f12e9888249ce7735dc395dd2c.svg" align="middle" width="127.92565499999999pt" height="24.65759999999998pt"/>

The crux in the derivation for the gradient relies on the Law of the Unconscious Statistician and can be found in section 5.2 of [this paper](https://arxiv.org/abs/1906.10652)

# Let's experiment
We now compare variance of either gradient estimator. Both gradients take an expectation that we approximate with Monte Carlo estimation. Appreciate that we make two approximations: one where the variational distribution approximates the posterior, another where the Monte Carlo samples approximate the expectations. 

The code implements both methods. When instantiating a `VIModel`, give as an argument which method you wish to run. The variances log in Tensorboard. So two runs with the pathwise derivative and the score function gradient look like:

![Comparing_grad_var](/home/rob/Dropbox/ml_projects/bbvi/bbvi/im/compare_var_grad.png)

Note on interpreting this figure: This figure shows estimator variance for different runs of the training. One cannot compare point to point and say that one estimator has lower variance, because they might concern different values of the variational parameters. In general, we can conclude, yes, that the pathwise derivative has lower variance.

# Compare Variational approximation with Laplace approximation
Laplace's method forms an alternative to the variational approximation. VI fits a full distribution to the posterior and uses some metric for comparison. In contrast, Laplace's method looks for the MAP solution, and makes a local approximation around the MAP parameter. Intuitively, VI seeks a good approximation, while Laplace seeks only a good parameter and makes an approximation afterwards.

The Hessian plays a central role in Laplace's method. At the MAP solution, we make a second order Taylor approximation to the log posterior. Assuming the first order derivative is zero at the MAP (which is the definition of the MAP, actually), now our Hessian is the inverse of a covariance matrix of an approximating Gaussian. In other words, we can treat the negative inverse of the Hessian as the covariance matrix. 

Let's consider the covariance matrix, <img alt="$\Sigma$" src="https://github.com/robromijnders/bbvi/blob/master/svgs/813cd865c037c89fcdc609b25c465a05.svg" align="middle" width="11.872245000000005pt" height="22.46574pt"/> of a Gaussian approximation. Let <img alt="$A^{-1} = \Sigma$" src="https://github.com/robromijnders/bbvi/blob/master/svgs/b7837d91b9f3877b36d9fb64147106b9.svg" align="middle" width="63.767055pt" height="26.76201000000001pt"/>, then we have for <img alt="$A$" src="https://github.com/robromijnders/bbvi/blob/master/svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg" align="middle" width="12.328800000000005pt" height="22.46574pt"/>:

<img alt="$A_{ij} = -\frac{\partial^2}{\partial w_i \partial w_j} [\log p(w|D) |_{w=w^*}]$" src="https://github.com/robromijnders/bbvi/blob/master/svgs/d6da1f63fc35b80e149691216126061c.svg" align="middle" width="235.37200499999997pt" height="33.459689999999995pt"/>

Our VI scheme uses a factorized Gaussian approximation. Therefore, we will only consider the diagonals in Laplace's method:    

<img alt="$\hat{\sigma}_{w_i} = (-\frac{\partial^2}{\partial w_i \partial w_i} [\log p(w|D) |_{w=w^*}])^{-1}$" src="https://github.com/robromijnders/bbvi/blob/master/svgs/1248b9314519f957ef286f75b23427a0.svg" align="middle" width="265.07695499999994pt" height="33.459689999999995pt"/>

Running this experiment, we get the following results: 
```bash
VI
  0.94,  2.24,  2.20,  0.63,  0.14,  2.24,  2.17,  2.34,  2.21,  2.03,  2.23,  2.10,  0.02,  2.27

Laplace
  0.14,  0.65,  0.79,  0.09,  0.02,  0.86,  0.79,  3.27,  1.22,  0.35,  1.87,  0.77,  0.00,  1.79

```

Each number indicates the standard deviation for the respective parameter. These numbers result from an experiment where the data was not standardized. 

Observations

  * The (relatively) higher and lower numbers coincide. This pattern confirms that both approximation capture that some parameters are more relevant than others.
  * In general, the deviations in the VI approximation are wider than in Laplace's approximation. This observation surprised me. VI is known to make compact approximations, relative to the posterior. While Laplace's approximation is known to have a wider estimate of the posterior (see, for example, the diagrams in Chapter 27 of Mackay's book). I need to research this observation more.
  
  
# Further reading

  * [Monte carlo gradient estimation in Machine Learning](https://arxiv.org/abs/1906.10652)
  * [Black box variational inference](https://arxiv.org/pdf/1401.0118.pdf)