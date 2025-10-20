
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import beta, norm, gamma

Writer = animation.PillowWriter
ext = "gif"


## I. Binomial Likelihood Distribution

### Initiate true distribution and our three prior distributions

# Form our "true" distribution from which we will sample data
true_p_theta = 0.3 # loaded coin, 30% heads
num_flips = 1000 # number of iid trials

# Three different hyperparameter cases for our prior distribution
params = [(1, 1), (10, 10), (7,3)]


### Make observations. Update our hyperparameters. Find posterior.

results = [] #Store all of our final results (parameter estimates and MSE) for each prior

for n, (alpha_param, beta_param) in enumerate(params):
    flip_history = [] # dataset of observations
    bayesian_history = [] # bayesian parameter estimate as we increment trials
    mse_b = [] # bayesian mse as we increment trials
    frequentist_history = [] # frequentise parameter estimate as we increment trials
    mse_f=[] #frequentist mse as we increment trials

    prior = beta(alpha_param, beta_param)

    # This is our live plotting for stretch goal #1
    plt.ioff()
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 1000)
    line, = ax.plot(x, prior.pdf(x))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(prior.pdf(x)) * 1.1) #autoscale ylim later
    ax.axvline(true_p_theta, color='red', linestyle='--', label='True p')
    ax.set_xlabel('p')
    ax.set_ylabel('Posterior PDF for Parameter p')
    ax.legend()
    filename = f'Project_1/Stretch_Goal_1_Vids/binomial_posterior_prior_{params[n][0]}_{params[n][1]}.{ext}'
    writer = Writer(fps=30)

    with writer.saving(fig, filename, dpi=150):
        for i in range(num_flips):
            flip = np.random.binomial(1, true_p_theta)
            flip_history.append(flip)

            # update
            if flip == 1:
                alpha_param += 1
            else:
                beta_param += 1

            # posterior calculation utilizing updated hyperparameters
            posterior = beta(alpha_param, beta_param)
            p_theta_b = posterior.mean()
            bayesian_history.append(p_theta_b)
            mse_b.append((p_theta_b - true_p_theta)**2)

            # max likelihood
            p_theta_f = flip_history.count(1) / (i+1)
            frequentist_history.append(p_theta_f)
            mse_f.append((p_theta_f - true_p_theta)**2)

            # update plot for this frame
            y = posterior.pdf(x)
            line.set_ydata(y)
            ax.set_title(
                f'CASE: BINOMIAL\nPrior: Beta({params[n]}); '
                f'Flip {i+1}: Beta({alpha_param}, {beta_param})'
            )
            # dynamic y-limit to avoid clipping as the posterior sharpens
            ymax = float(np.max(y)) * 1.1
            ax.set_ylim(0, max(ymax, 1.0))

            # write this frame
            writer.grab_frame()

    result = {"prior": params[n], "bayesian estimate": p_theta_b, "frequentist estimate": p_theta_f, "bayesian mse": mse_b, "frequentist mse": mse_f}
    results.append(result)


### Final results

for result in results:
    plt.ioff()
    plt.close()
    print(f"\nPrior Hyperparameters: {result["prior"]}. Final Bayesian Estimate: {result["bayesian estimate"]:.4f}. Final ML Estimate: {result["frequentist estimate"]:.4f}")

    plt.figure()
    plt.xlim(0, 50)
    plt.plot(result["bayesian mse"], label="Bayesian MSE")
    plt.plot(result["frequentist mse"], label="Frequentist MSE")
    plt.title(f"CASE: BINOMIAL \nMSE vs. Number of Flips for the Beta{result["prior"]} Prior Distribution" )
    plt.xlabel("Number of Flips")
    plt.ylabel("MSE for Parameter p")
    plt.legend()
    plt.show()