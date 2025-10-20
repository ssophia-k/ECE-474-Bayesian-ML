import numpy as np
import matplotlib.pyplot as plt


lower_x_bound, upper_x_bound = 0, 1
true_freq = 2 * np.pi
true_noise_std = 0.1  # This is the noise that gets added when we draw a target observation
true_noise_var = true_noise_std**2
true_noise_beta = 1 / true_noise_var

# Basis functions: 9 Gaussian basis functions
# The mu_j are spatial locations of the basis functions
# We will later space them evenly (as in the example in the book)
def phi(x, mu_j, s=0.1):  # s parameter unimportant since these will get scaled by weights later
    return np.exp(-0.5 * ((x - mu_j)/s)**2)

"""### Initiate our prior"""

mu_0 = np.zeros(9)  # initial guesses for [w0, w1, ..., w8]
alpha_0 = 2.0       # single prior precision for both weights
s_0 = (1/alpha_0) * np.eye(9)  # prior covariance matrix. assume uncorrelated, single variance

"""
INITIALIZATIONS BEFORE LOOP
"""
num_observations = 1000
x_obs_history, t_obs_history = [], []  # store x and target var t
phi_x = []  # Design matrix rows for 9 Gaussian basis functions; each row = [phi_1(x), ..., phi_9(x)]

# Fixed basis locations and x-grid for plotting
mu_js = np.linspace(lower_x_bound, upper_x_bound, 9)
x_plot = np.linspace(lower_x_bound, upper_x_bound, 400)

# Current posterior parameters (start at prior)
mu_n, s_n = mu_0.copy(), s_0.copy()

# Figure and which Ns to snapshot
snapshot_Ns = [1, 2, 4, 25]
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axes = axes.ravel()

"""
START OUR OBSERVATIONS LOOP. OBSERVE, PLOT, CALCULATE, ETC
"""
for n in range(num_observations ):

    # Sample an x observation
    x_obs = np.random.uniform(lower_x_bound, upper_x_bound)

    # Generate noisy target t = sin(frequency * x) + ε
    true_y = np.sin(true_freq * x_obs)
    t_obs = true_y + np.random.normal(0.0, true_noise_std)

    # Append to histories
    x_obs_history.append(x_obs)
    t_obs_history.append(t_obs)

    # Design matrix row for this x
    phi_x.append([phi(x_obs, mu_j) for mu_j in mu_js])

    # Convert our lists to arrays for linalg calcs
    Phi = np.asarray(phi_x)  # nx9
    t_vec = np.asarray(t_obs_history)  # n,

    # Update our prior parameters to get our posterior
    s_n_inv = np.linalg.inv(s_0) + true_noise_beta * (Phi.T @ Phi)  # 9x9
    s_n = np.linalg.inv(s_n_inv)  # 9x9
    mu_n = true_noise_beta * (s_n @ Phi.T @ t_vec)  # 9,

    # Now predictive distribution for plotting on the same grid as the plot
    sigma_n = []
    mean_pred = []
    for xi in x_plot:
        phi_x_n = np.array([phi(xi, mu_j) for mu_j in mu_js])  # 9,
        sigma_n.append(np.sqrt(1/true_noise_beta + phi_x_n.T @ s_n @ phi_x_n))
        mean_pred.append(mu_n.T @ phi_x_n)
    mean_pred = np.asarray(mean_pred).ravel()
    sigma_n = np.asarray(sigma_n).ravel()

    # Snapshot and plot at desired N
    N_now = len(t_obs_history)
    if N_now in snapshot_Ns:
        ax = axes[snapshot_Ns.index(N_now)]
        # true function
        ax.plot(x_plot, np.sin(true_freq * x_plot), color='green', linewidth=1.5)
        # predictive mean
        ax.plot(x_plot, mean_pred, color='red', linewidth=2)
        # ±1σ band
        ax.fill_between(x_plot, mean_pred - sigma_n, mean_pred + sigma_n,
                        color='red', alpha=0.25)
        # observations used so far
        ax.scatter(x_obs_history[:N_now], t_obs_history[:N_now],
                   facecolors='none', edgecolors='blue')

        ax.set_title(f"N = {N_now}")
        ax.set_xlim(lower_x_bound, upper_x_bound)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_yticks([-1, 0, 1])

# labels layout
for ax in axes[2:]:
    ax.set_xlabel("x")
for ax in axes[::2]:
    ax.set_ylabel("t")

fig.suptitle("Bayesian Linear Regression with 9 Gaussian Basis Functions", y=0.95, fontsize=12)
fig.tight_layout()
fig.subplots_adjust(top=0.90)
plt.show()
