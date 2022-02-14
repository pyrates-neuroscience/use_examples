from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

# preparations
##############

# plot settings
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 400
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 8.0
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 0.7
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
idx_c = target_params['source_idx_0']
idx_r = target_params['target_idx_0']

for row, col, w, w_t in zip(idx_c, idx_r, target_params['weight'], fitted_data["weight"]):
    C[row-1, col-1] = w
    C_t[row-1, col-1] = w_t

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=4, figure=fig)

# plot matrices
for i, mat in enumerate([C, C_t]):
    ax = fig.add_subplot(grid[i, 0])
    im = ax.imshow(mat, aspect='equal')
    ax.set_xlabel(r'LI #')
    ax.set_ylabel(r'LI #')


# plot signals
for i, sig in enumerate([fitted_data["y"], target_signal]):
    ax = fig.add_subplot(grid[i, 1:])
    ax.plot(sig)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('rate')

# finishing touches
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)
fig.canvas.draw()
fig.savefig('leaky_integrator.pdf')
plt.show()
