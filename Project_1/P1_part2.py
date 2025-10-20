import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

## II. Gaussian Likelihood Distribution with Known Variance

# True data-generating distribution
true_mean = 0
true_sigma = 1
true_variance = true_sigma**2
num_samples = 1000

# Priors: (mu0, tau0) where tau0 is the prior std dev
params = [(0, 1), (0, 10), (5, 5)]

results = []  # final estimates + MSE for each prior

# Choose a movie writer: MP4 if ffmpeg exists, else GIF
use_ffmpeg = animation.writers.is_available('ffmpeg')
Writer = animation.FFMpegWriter if use_ffmpeg else animation.PillowWriter
ext = "mp4" if use_ffmpeg else "gif"

for n, (mean_param, tau_param) in enumerate(params, start=1):
    # Working copies (posterior starts at the prior)
    mu_posterior = float(mean_param)
    tau_posterior_var = float(tau_param**2)  # variance
    tau_posterior = float(tau_param)         # std dev

    sample_history = []
    bayesian_history = []
    frequentist_history = []
    mse_b = []
    mse_f = []

    # Figure for the movie (not shown interactively)
    plt.ioff()
    fig, ax = plt.subplots()
    x = np.linspace(-5, 5, 1000)
    prior_pdf = norm(mean_param, tau_param).pdf(x)
    line, = ax.plot(x, prior_pdf)
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, max(prior_pdf) * 1.1)
    ax.axvline(true_mean, color='red', linestyle='--', label='True mean')
    ax.set_ylabel('Posterior PDF for MEAN')
    ax.legend(loc='upper right')

    filename = f'Project_1/Stretch_Goal_1_Vids/gaussian_knownvar_prior_{mean_param}_{tau_param}.{ext}'
    writer = Writer(fps=30)

    with writer.saving(fig, filename, dpi=150):
        for i in range(num_samples):
            # Observe one sample
            sample = np.random.normal(true_mean, true_sigma)
            sample_history.append(sample)

            # Conjugate update with one new datum:
            # 1/Var_new = 1/Var_old + 1/sigma^2
            # mu_new = (mu_old/Var_old + x/sigma^2) / (1/Var_old + 1/sigma^2)
            inv_var_old = 1.0 / tau_posterior_var
            inv_var_data = 1.0 / true_variance
            inv_var_new = inv_var_old + inv_var_data
            tau_posterior_var = 1.0 / inv_var_new
            mu_posterior = (mu_posterior * inv_var_old + sample * inv_var_data) / inv_var_new
            tau_posterior = np.sqrt(tau_posterior_var)

            # Posterior distribution, estimates, and MSE
            posterior = norm(mu_posterior, tau_posterior)
            bayes_est = mu_posterior
            ml_est = np.mean(sample_history)

            bayesian_history.append(bayes_est)
            frequentist_history.append(ml_est)
            mse_b.append((bayes_est - true_mean) ** 2)
            mse_f.append((ml_est - true_mean) ** 2)

            # Update plot and write frame
            y = posterior.pdf(x)
            line.set_ydata(y)
            ax.set_title(
                f'CASE: GAUSSIAN (Known Variance)\n'
                f'Prior: N({mean_param}, {tau_param}^2); '
                f'Sample {i+1}: N({mu_posterior:.4f}, {tau_posterior:.4f}^2)'
            )
            ax.set_ylim(0, max(float(np.max(y)) * 1.1, 0.2))
            writer.grab_frame()

    plt.close(fig)  # free memory

    results.append({
        "prior": (mean_param, tau_param),
        "bayesian estimate": bayes_est,
        "frequentist estimate": ml_est,
        "bayesian mse": mse_b,
        "frequentist mse": mse_f
    })

# ---- Final results: print + DISPLAY MSE plots ----
for r in results:
    print(
        f"\nPrior Hyperparameters: {r['prior']}. "
        f"Final Bayesian Estimate: {r['bayesian estimate']:.4f}. "
        f"Final ML Estimate: {r['frequentist estimate']:.4f}"
    )

    plt.figure()
    plt.xlim(0, 50)
    plt.plot(r["bayesian mse"], label="Bayesian MSE")
    plt.plot(r["frequentist mse"], label="Frequentist MSE")
    plt.title(
        f"CASE: GAUSSIAN KNOWN VAR\n"
        f"MSE vs. Number of Trials for the N{r['prior']} Prior Distribution"
    )
    plt.xlabel("Number of Trials")
    plt.ylabel("MSE for MEAN")
    plt.legend()
    plt.show()
