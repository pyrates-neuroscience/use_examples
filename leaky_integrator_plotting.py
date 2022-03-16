from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

# preparations
##############

import os
p = os.environ['PATH']
# plot settings
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 400
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams["font.family"] = "sans-serif"

# load data
target_params = np.load("li_params.npz")
target_signal = np.load("li_target.npy")
fitted_data = np.load("li_fitted_o.npz")

# reconstruct connectivity matrices
###################################

N = len(target_params["r0"])
C = np.zeros((N, N))
C_t = np.zeros_like(C)
for i, (w_t, w) in enumerate(zip(target_params['weight'], fitted_data["weight"])):
    C[i, :] = w[i::N]
    C_t[i, :] = w_t[i::N]

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=9, figure=fig)

# plot raw matrices
for i, (mat, subplot, cond) in enumerate(zip([C, C_t], ['A', 'C'], ['Fitted', 'Target'])):
    ax = fig.add_subplot(grid[i, :3])
    im = ax.imshow(mat, aspect='equal')
    ax.set_xlabel('LI index')
    ax.set_ylabel('LI index')
    fig.colorbar(im, ax=ax, shrink=0.6)
    plt.title(fr"\textbf{{({subplot})}} {cond} connectivity matrix")

# plot signals
for i, (sig, subplot, cond) in enumerate(zip([fitted_data["y"], target_signal], ["B", "D"], ["Fitted", "Target"])):
    ax = fig.add_subplot(grid[i, 3:])
    ax.plot(sig)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('rate (x)')
    ax.set_title(fr"\textbf{{({subplot})}} {cond} network signal")

# finishing touches
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)
fig.canvas.draw()
fig.savefig('leaky_integrator.pdf')
plt.show()
