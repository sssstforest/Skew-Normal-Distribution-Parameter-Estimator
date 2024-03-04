import SNParameterEstimator
from SNParameterEstimator import ParameterEstimator

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm
from scipy import special

# Define the number of samples, the location, the scale, and the skewness
num_samples = 10000
mu_data = 1.0  # location (mean) 
sd_data = 5.0  # scale (standard deviation) 
alpha_data = 5.0  # skewness (negative value means left skew, positive means right skew)

# Generate the skew-normal dataset
dist = skewnorm(alpha_data, mu_data, sd_data)
samples = dist.rvs(num_samples)

# Create a histogram of the samples
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
p = skewnorm.pdf(x, alpha_data, mu_data, sd_data)
plt.plot(x, p, 'k', linewidth=2)

x_values = x  # x values are the points at which you're evaluating the PDF
y_values = p  # y values are the corresponding values of the PDF

mu, sd, alpha = ParameterEstimator(x_values, y_values, 501)

print("Estimated location (mean): ", mu)
print("Estimated scale (standard deviation): ", sd)
print("Estimated skewness: ", alpha)