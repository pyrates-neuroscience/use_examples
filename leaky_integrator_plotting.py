from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

# preparations
##############

# plot settings
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
fitted_data = np.load("li_fitted.npz")

# reconstruct connectivity matrices
###################################

N = len(target_params["r0"])
C = np.zeros((N, N))
C_t = np.zeros_like(C)
idx_c = target_params['source_idx_0'] - 1
idx_r = target_params['target_idx'] - 1

for row, col, w_t, w in zip(idx_r, idx_c, target_params['weight'], fitted_data["weight"]):
    C[row, col] = w
    C_t[row, col] = w_t

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=9, figure=fig)

# plot raw matrices
for i, mat in enumerate([C, C_t]):
    ax = fig.add_subplot(grid[i, :2])
    im = ax.imshow(mat, aspect='equal')
    ax.set_xlabel(r'LI #')
    ax.set_ylabel(r'LI #')
    fig.colorbar(im, ax=ax, shrink=0.6)
    if i == 0:
        plt.title(r"\textbf{(A)} Connectivity matrices")

# plot sum of rows
for i, mat in enumerate([C, C_t]):
    ax = fig.add_subplot(grid[i, 2])
    im = ax.imshow(np.sum(mat, axis=1, keepdims=True), aspect='equal')
    ax.set_ylabel(r'LI #')
    fig.colorbar(im, ax=ax, shrink=0.6)
    if i == 0:
        plt.title(r"\textbf{(B)} Summed inputs")

# plot signals
for i, sig in enumerate([fitted_data["y"], target_signal]):
    ax = fig.add_subplot(grid[i, 3:])
    ax.plot(sig)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('rate (x)')
    if i == 0:
        plt.title(r"\textbf{(C)} Network signals")

# finishing touches
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)
fig.canvas.draw()
fig.savefig('leaky_integrator.pdf')
plt.show()
