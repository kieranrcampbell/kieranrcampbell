---
layout: post
title: "Fast vectorized sampling from truncated normal distributions in python"
tags:
    - python
    - notebook
---
Recently I've been playing a lot with MCMC sampling and in particular the
Metropolis-Hastings algorithm. MH requires a proposal distribution that can
generate subsequent samples for the next state in the markov chain, eventually
filling out the posterior distribution.

Typically a Gaussian distribution centred around the currently value is used as
the proposal distribution. However, sometimes the support of the target
distribution isn't the entire real line (e.g. the mean of a negative binomial
distribution must be positive). In this case, sampling from a truncated normal
distribution with lower bound 0 makes a lot of sense. Coding in python, I used
the `truncnorm` class from `scipy.stats` to sample:

**In [1]:**

{% highlight python %}
from scipy.stats import truncnorm
import numpy as np

np.random.seed(123)

lower_clip = 0.
upper_clip = np.inf
mu = 3.
sd = 2.

truncnorm.rvs((lower_clip - mu) / sd, (upper_clip - mu) / sd, mu, sd)
{% endhighlight %}

 


    4.1464110604259821



So far so good. However, `truncnorm.rvs` isn't vectorised for different `mu`,
which comes in really handy in regression settings such as negative binomial
regression.

No matter, we can write a wrapper function and use `np.vectorize`:

**In [2]:**

{% highlight python %}
def truncnorm_rvs_wrapper(mu, sd, lower_clip, upper_clip):
    return truncnorm.rvs((lower_clip - mu) / sd, upper_clip, mu, sd)
    
truncnorm_rvs = np.vectorize(truncnorm_rvs_wrapper, otypes=[np.float])
{% endhighlight %}

Now we can sample for a whole range of `mu`:

**In [3]:**

{% highlight python %}
mus = np.random.randint(1,10,50)
truncnorm_rvs(mus, sd, lower_clip, upper_clip)[1:10]
{% endhighlight %}

 


    array([ 1.95424841,  6.9155242 ,  6.50475593,  4.15263574,  7.57421282,
            0.71420649,  3.34918575,  2.85799086,  1.96588436])



However, compared to sampling from the usual Gaussian distribution, this is
slow. Like, _really_ slow:

**In [4]:**

{% highlight python %}
%timeit truncnorm_rvs(mus, sd, lower_clip, upper_clip)
%timeit np.random.normal(mus, sd)
{% endhighlight %}

    100 loops, best of 3: 1.27 ms per loop
    100000 loops, best of 3: 10.9 µs per loop


While 1.27ms doesn't seem like much, when this gets evaluated thousands of times
per sample it can really start to hold things up and is almost two orders of
magnitude slower than sampling from a normal distribution. So the question
becomes, can we implement truncated normal sampling that is almost as fast as
normal normal sampling?

Consider the following function:

**In [5]:**

{% highlight python %}
def truncnorm_rvs_recursive(x, sigma, lower_clip):
    q = np.random.normal(x, sigma, size=len(x))
    if np.any(q < lower_clip):
        q[q < lower_clip] = truncnorm_rvs_recursive(x[q < lower_clip], sigma, lower_clip)
{% endhighlight %}

**In [6]:**

{% highlight python %}
%timeit truncnorm_rvs_recursive(mus, sd, lower_clip)
{% endhighlight %}

    10000 loops, best of 3: 52.8 µs per loop


And just like that, we've improved on the stock `truncnorm.rvs` by a factor of
around 50. While we haven't implemented an upper bound, it could easily be
extended to such a scenario (though would probably trigger further recursions).

One of the problems with such an approach is that the number of recursions is
itself probabilistic and data-dependent - the number of recursions will depend
on the number of samples generated each time that lie outside `lower_clip`. Can
we put a bound on the expectation of the number of recursions? We know sampling
from the standard normal is around 100 time faster than `truncnorm.rvs`, so as
long as the number of recursions is less than 100 our method is still faster.

For the distribution with only a `lower_clip`, the probability a sample will be
generated outside the plausible range is given by \\(Pr(\mathrm{outside}) =
\Phi(L)\\) for a lower bound \\(L\\) where \\(\Phi\\) is the cumulative normal
distribution.

Say we take the rather extreme situation where the means of the truncated
distribution are all the same as `lower_clip`. In this case, the probability of
a sample being rejected is 0.5 and so the expected number of elements that will
fall outside the allowed range is \\(N/2\\) for the mean vector of length
\\(N\\). Then the expected number of elements falling outside on the
\\(i^{th}\\) recursion is given by \\( \frac{N}{2^i} \\). We expect this
function to stop recursing at recursion \\(m\\) when \\(\frac{N}{2^m} < 0.5\\)
(this is of course approximate since we could continue recursing for a further
\\(p\\) recursions with probability \\(0.5^p\\)). Solving for \\(m\\) we get

$$ m = log_2 N + 1 $$

In other words, the _expected_ number of recursions scales logarithmically in
\\(N\\), making it particularly promising as a competitor (at least for lower
bounded) truncated normal sampling in `python`.

A full IPython notebook can be found [here](http://nbviewer.ipython.org/github/kieranrcampbell/blog-notebooks/blob/master/Fast%20vectorized%20sampling%20from%20truncated%20normal%20distributions%20in%20python.ipynb).
