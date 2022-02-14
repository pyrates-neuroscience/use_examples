from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

# preparations
##############

# plot settings
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 400
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 0.7
plt.rcParams["font.family"] = "cmss"

# load data
target_params = np.load("li_params.npz")
target_signal = np.load("li_target.npy")
fitted_data = np.load("li_fitted.npz")

# reconstruct connectivity matrices
###################################


# plotting
##########

