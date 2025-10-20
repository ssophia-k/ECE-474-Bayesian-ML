import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import multivariate_normal

"""## I. Linear True Relationship
Model: $y = w_0 + w_1x$

### Initiate true relationship and noise
"""

lower_x_bound, upper_x_bound = -1, 1
true_slope, true_intercept = 2.0, 1.0
true_noise_std = 0.2 # This is the noise that gets added when we draw a target observation
true_noise_var = true_noise_std**2
true_noise_beta = 1 / true_noise_var

"""### Initiate our prior"""

mu_0 = np.array([0.0, 0.0]) # initial guesses for [w0, w1]
alpha_0 = 2.0 # single prior precision for both weights
s_0 = (1/alpha_0) * np.eye(2) # prior covariance matrix. assume uncorrelated, single variance

"""### Start drawing observations and updating

Note that to plot the likelihood for single observation over weight-space:

$P(t_n | w) = N(t_n | w^T \phi(x_n), \beta^{-1}) =
\frac{1}{\sqrt{2\pi\beta^{-1}}} e^{ \frac{-\beta}{2} (t_n - w^T \phi(x_n) )^2} = \frac{1}{\sqrt{2\pi\beta^{-1}}} e^{ \frac{-\beta}{2} (t_n - (w_0\phi_0(x_n) + w_1\phi_1(x_n) )^2 }
= \frac{1}{\sqrt{2\pi\beta^{-1}}} e^{ \frac{-\beta}{2} (t_n - (w_0 + w_1 x_n ))^2 }$
"""

"""
PLOTTING SETUP STUFF
"""
# Weight-space grid bounds. Chosen for visibility only.
w0_grid = np.linspace(-1.5, 3.0, 200)
w1_grid = np.linspace(-1.5, 3.0, 200)
W0, W1 = np.meshgrid(w0_grid, w1_grid)  # each is shape (G, G)

# Which update steps to visualize (rows of the figure)
steps = [0, 1, 2, 5, 20]

# One row per step; columns: [likelihood of latest obs | prior/posterior | data space]
fig, axes = plt.subplots(len(steps), 3,
                         figsize=(10, 1.6*len(steps)), # width;height per row
                         constrained_layout=True)
fig.suptitle(f"Bayesian Linear Regression Updates for {steps} observations", fontsize=12)
# Column titles at the top
col_titles = ["Current likelihood", "Prior/posterior ", "Data space"]
for j, title in enumerate(col_titles):
    axes[0, j].set_title(title, fontsize=12)



"""
INITIALIZATIONS BEFORE LOOP
"""
num_observations = 1000
x_obs_history, t_obs_history = [], [] # store x and target var t
phi_x = []  # Design matrix initialization. Simple linear model. rows: [1, x]

# Current posterior parameters (start at prior)
mu_n, s_n = mu_0.copy(), s_0.copy()

row = 0 # for plotting, since we will have multiple rows of results



"""
START OUR OBSERVATIONS LOOP. OBSERVE, PLOT, CALCULATE, ETC
"""
for n in range(num_observations + 1):

    # Visualization only at selected steps
    if n in steps:

        # LEFT COLUMN: likelihood of the most recent observation in weight space.
        ax = axes[row, 0]
        if n == 0:
            # No data for likelihood yet. Leave blank for the "prior-only" row
            ax.axis('off')
        else:
            xn, tn = x_obs_history[-1], t_obs_history[-1] # Take most recent observations
            resid = tn - (W0 + W1 * xn) # W0 and W1 are the gridded weights (for plotting, not real weight values)
            Likelihood = (2 * np.pi * true_noise_beta**(-1))**(-0.5) * np.exp(-0.5 * true_noise_beta * resid**2)
            ax.imshow(Likelihood,
                      extent=[w0_grid[0], w0_grid[-1], w1_grid[0], w1_grid[-1]],
                      origin='lower', aspect='auto', cmap='plasma')
            ax.set_xlabel("w0"); ax.set_ylabel("w1")

        # MIDDLE COLUMN: prior (n=0) or current posterior (n>0) in weight space.
        ax = axes[row, 1]
        mean = mu_0 if n == 0 else mu_n
        cov  = s_0  if n == 0 else s_n
        mvn = multivariate_normal(mean=mean, cov=cov)
        # Evaluate the 2D Gaussian density at each (w0, w1) grid point
        Posterior = mvn.pdf(np.dstack([W0, W1]))
        ax.imshow(Posterior,
                  extent=[w0_grid[0], w0_grid[-1], w1_grid[0], w1_grid[-1]],
                  origin='lower', aspect='auto', cmap='plasma')
        # Mark the true parameter location for reference
        ax.plot(true_intercept, true_slope, "w+", ms=5, mew=1, label = "True (w0, w1)")
        ax.legend(loc='upper left', fontsize=6)
        ax.set_xlabel("w0"); ax.set_ylabel("w1")

        # Right column: data space (x,y). Sample 6 weights from current prior/posterior
        # to show how predicted lines converge as uncertainty shrinks.
        ax = axes[row, 2]
        x_plot = np.linspace(-1, 1, 200)
        mean_plot = mu_0 if n == 0 else mu_n
        cov_plot  = s_0  if n == 0 else s_n
        W_samples = np.random.multivariate_normal(mean_plot, cov_plot, size=6)
        for w0, w1 in W_samples:
            ax.plot(x_plot, w0 + w1 * x_plot, lw=1)
        if n > 0:
            # Show all observed points so far
            ax.scatter(x_obs_history, t_obs_history, facecolors='none', edgecolors='C0')
        ax.set_xlim(-1, 1); ax.set_ylim(-1.5, 3.5)
        ax.set_xlabel("x"); ax.set_ylabel("y")

        row += 1
        if row == len(steps):
            # Once we reach the number of plots we want, we can break this loop
            break

    if n == num_observations:
        # Exit loop once we reach tot number of observations we want
        break

    # Here we actually do the math to update our distributions per observation

    # Sample an x observ ation
    x_obs = np.random.uniform(lower_x_bound, upper_x_bound)

    # Generate noisy target t = (true_intercept + true_slope * x) + ε
    true_y = true_slope * x_obs + true_intercept
    t_obs = true_y + np.random.normal(0.0, true_noise_std)

    # Append to histories and design matrix
    x_obs_history.append(x_obs)
    t_obs_history.append(t_obs)
    phi_x.append([1.0, x_obs]) # There will only be 2 basis functions per observation: phi_0(x)=1, phi_1(x)=x

    # Convert our lists to arrays for linalg calcs
    Phi = np.asarray(phi_x) # nx2
    t_vec = np.asarray(t_obs_history) # nx1

    # Update our prior parameters to get our posterior
    s_n_inv = np.linalg.inv(s_0) + true_noise_beta * (Phi.T @ Phi) # 2x2
    s_n = np.linalg.inv(s_n_inv) # 2x2
    mu_n = true_noise_beta * (s_n @ Phi.T @ t_vec) #2x1

print(f"After {num_observations} iterations, our model is y = {mu_n[0]} + {mu_n[1]}x")
plt.show()

