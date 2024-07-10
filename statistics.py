import math
import scipy.stats as stats

# Conditional distribution of Random Variables
def conditional_distribution_continuous(f_xy, a, b, y):
    return integrate.quad(lambda z: f_xy(a, z), b, y)

def conditional_distribution_discrete(f_xy, a, b, x):
    return sum(f_xy(a, y) for y in range(b, x+1))

# Bayes' Theorem
def bayes_theorem(f_xy, f_y):
    return f_xy / f_y

# Conditional Expectation
def conditional_expectation_discrete(E_X_given_Y):
    return sum(E_X_given_Y)

def conditional_expectation_continuous(f_xy, a, b):
    return integrate.quad(lambda x: x * f_xy(a, x), b)

# Conditional Variance
def conditional_variance(E_X_squared_given_Y, E_X_given_Y_squared):
    return E_X_squared_given_Y - E_X_given_Y_squared

# Joint Expectation (Discrete)
def joint_expectation_discrete(X, f_xy):
    return sum(xi * f_xy for xi in X)

# Covariance
def covariance(E_XY, E_X_E_Y):
    return E_XY - E_X_E_Y

# Discrete Distributions
def uniform_distribution(n):
    return 1/n

def binomial_distribution(n, p, x):
    return math.comb(n, x) * (p ** x) * ((1 - p) ** (n - x))

def bernoulli_distribution(p, x):
    return p if x == 1 else 1 - p

def hypergeometric_distribution(N, M, n, k):
    return math.comb(M, k) * math.comb(N - M, n - k) / math.comb(N, n)

def poisson_distribution(lmbda, x):
    return (lmbda ** x) * math.exp(-lmbda) / math.factorial(x)

def geometric_distribution(p, x):
    return p * ((1 - p) ** (x - 1))

def negative_binomial_distribution(k, p, x):
    return math.comb(x - 1, k - 1) * (p ** k) * ((1 - p) ** (x - k))

# Continuous Distributions
def normal_distribution(mu, sigma, x):
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def standard_normal_distribution(x):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x ** 2)

def normal_probability(mu, sigma, x0):
    return stats.norm.cdf(x0, mu, sigma)

# Inductive Statistics
def true_mean(X):
    return sum(X) / len(X)

def sample_variance(X):
    mean = true_mean(X)
    return sum((xi - mean) ** 2 for xi in X) / (len(X) - 1)

def proportion(k, n):
    return k / n

def confidence_interval_mean(mu, u, sigma, n):
    return (mu - u * (sigma / math.sqrt(n)), mu + u * (sigma / math.sqrt(n)))

# Usage example (values are for demonstration and should be replaced with actual data)
X = [1, 2, 3, 4, 5]
mu = true_mean(X)
sigma = math.sqrt(sample_variance(X))
n = len(X)
confidence_interval = confidence_interval_mean(mu, 1.96, sigma, n)

print("True Mean:", mu)
print("Sample Variance:", sigma)
print("Confidence Interval:", confidence_interval)
