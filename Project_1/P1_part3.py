import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import gamma

## III. Gaussian Likelihood Distribution with Known Mean

true_mean = 0
true_sigma = 1
true_variance = true_sigma**2
true_precision = 1 / true_variance
num_samples = 1000

# Priors for precision τ ~ Gamma(a, b) with rate b (scale = 1/b)
params = [(0.1, 0.1), (1, 1), (4, 6)]

results = []

# Choose a movie writer: MP4 if ffmpeg exists, else GIF
use_ffmpeg = animation.writers.is_available('ffmpeg')
Writer = animation.FFMpegWriter if use_ffmpeg else animation.PillowWriter
ext = "mp4" if use_ffmpeg else "gif"

for (a0, b0) in params:
    a_param_new = float(a0)
    b_param_new = float(b0)

    sample_history = []
    bayesian_history = []
    frequentist_history = []
    mse_b = []
    mse_f = []

    # Figure for the movie (not shown interactively)
    plt.ioff()
    fig, ax = plt.subplots()

    # Start x-axis for Gamma(·) on [0, ∞)
    x = np.linspace(0, 10, 1000)
    prior_pdf = gamma.pdf(x=x, a=a0, scale=1.0/b0)
    line, = ax.plot(x, prior_pdf)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(prior_pdf) * 1.1 if np.isfinite(np.max(prior_pdf)) and np.max(prior_pdf) > 0 else 1.0)
    ax.axvline(true_precision, color='red', linestyle='--', label='True precision')
    ax.set_xlabel('Precision τ')
    ax.set_ylabel('Posterior PDF for PRECISION')
    ax.legend(loc='upper right')

    filename = f'Project_1/Stretch_Goal_1_Vids/gaussian_knownmean_prior_{a0}_{b0}.{ext}'
    writer = Writer(fps=30)

    with writer.saving(fig, filename, dpi=150):
        ssq = 0.0  # running sum of squared deviations (mean known)
        for i in range(num_samples):
            sample = np.random.normal(true_mean, true_sigma)
            sample_history.append(sample)
            ssq += (sample - true_mean) ** 2

            # Conjugate update for τ | data ~ Gamma(a + n/2, b + 0.5 * sum (x-μ)^2)
            a_param_new += 0.5
            b_param_new += 0.5 * (sample - true_mean) ** 2

            a_posterior = a_param_new
            b_posterior = b_param_new

            # Posterior mean of precision: E[τ | data] = a' / b'
            bayes_prec = a_posterior / b_posterior
            bayesian_history.append(bayes_prec)

            # ML estimate of precision with known mean: τ_ML = n / sum (x-μ)^2
            ml_prec = (i + 1) / ssq
            frequentist_history.append(ml_prec)

            # MSE wrt true precision
            mse_b.append((bayes_prec - true_precision) ** 2)
            mse_f.append((ml_prec - true_precision) ** 2)

            # Update curve and axes dynamically
            y = gamma.pdf(x=x, a=a_posterior, scale=1.0/b_posterior)
            line.set_ydata(y)
            ax.set_title(
                f'CASE: GAUSSIAN KNOWN MEAN\n'
                f'Prior: Gamma(a={a0}, b={b0}); '
                f'Sample {i+1}: Gamma(a={a_posterior:.4f}, b={b_posterior:.4f})'
            )

            # Auto-scale y; expand x-range if mass creeps beyond 10
            ymax = float(np.max(y)) if np.isfinite(np.max(y)) else 1.0
            ax.set_ylim(0, max(0.2, ymax * 1.1))
            # If the 99.9th percentile is beyond current x max, extend it
            from math import isfinite
            try:
                q = gamma(a=a_posterior, scale=1.0/b_posterior).ppf(0.999)
                if isfinite(q) and q > ax.get_xlim()[1] * 0.95:
                    new_max = min(50.0, q * 1.2)
                    x = np.linspace(0, new_max, 1200)
                    line.set_xdata(x)
                    ax.set_xlim(0, new_max)
                    y = gamma.pdf(x=x, a=a_posterior, scale=1.0/b_posterior)
                    line.set_ydata(y)
            except Exception:
                pass

            writer.grab_frame()

    plt.close(fig)

    results.append({
        "prior": (a0, b0),
        "bayesian estimate": bayes_prec,
        "frequentist estimate": ml_prec,
        "bayesian mse": mse_b,
        "frequentist mse": mse_f
    })

# ---- Final results: print + DISPLAY MSE plots ----
for r in results:
    print(
        f'\nPrior Hyperparameters: Gamma{r["prior"]}. '
        f'Final Bayesian Precision: {r["bayesian estimate"]:.4f}. '
        f'Final ML Precision: {r["frequentist estimate"]:.4f}'
    )

    plt.figure()
    plt.xlim(0, 50)
    plt.plot(r["bayesian mse"], label="Bayesian MSE")
    plt.plot(r["frequentist mse"], label="Frequentist MSE")
    plt.title(
        f'CASE: GAUSSIAN KNOWN MEAN\n'
        f'MSE vs. Number of Trials for the Gamma{r["prior"]} Prior (Precision)'
    )
    plt.xlabel("Number of Trials")
    plt.ylabel("MSE for PRECISION")
    plt.legend()
    plt.show()
