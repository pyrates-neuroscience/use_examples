import pickle

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
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['font.size'] = 14.0
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams["font.family"] = "sans-serif"

# load data
data = pickle.load(open("leaky_integrator_data.pkl", "rb"))

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=3, figure=fig)

# plot loss landscape
n = 50
ticks = np.arange(0, n, 10)
labels = np.round(10.0**np.linspace(-1.0, 1.0, num=n)[ticks], decimals=1)
ax = fig.add_subplot(grid[:, 0])
im = ax.imshow(np.log(data["loss_landscape"]), aspect='equal')
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$k$')
ax.set_xticks(ticks, labels=[str(l) for l in labels])
ax.set_yticks(ticks, labels=[str(l) for l in labels])
ax.invert_yaxis()
c = fig.colorbar(im, ax=ax, shrink=0.6, label="log(MSE)")
plt.title(r"\textbf{(A)} 2D loss landscape")

# plot training progress
n = 50
vals = 10.0**np.linspace(-1.0, 1.0, num=n)
rows, cols = [], []
for tau, k in zip(data["taus"], data["Js"]):
    rows.append(np.argmin(np.abs(vals-k)))
    cols.append(np.argmin(np.abs(vals-tau)))
plt.plot(cols, rows, marker="o", color="white", markersize=3)

# plot signals
dt = 1e-3
time = np.arange(0, len(data["predictions"]))*dt
for i, (s, t) in enumerate(zip([data["predictions"], data["targets"]],
                               [r"\textbf{(B)} predicted network signal", r"\textbf{(C)} target network signal"])):
    ax = fig.add_subplot(grid[i, 1:])
    ax.plot(time, s)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel(r'$u$')
    ax.set_title(t)

# finishing touches
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)
fig.canvas.draw()
fig.savefig('leaky_integrator.pdf')
plt.show()
