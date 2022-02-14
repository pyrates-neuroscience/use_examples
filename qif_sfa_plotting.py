from matplotlib import gridspec
import matplotlib.pyplot as plt
from pyauto import PyAuto
import pickle

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
a = PyAuto.from_file("qif_sfa_data.pkl", auto_dir='/home/rgast/PycharmProjects/auto-07p')

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

# time continuation
ax = fig.add_subplot(grid[0, 1])
a.plot_continuation('PAR(14)', 'U(1)', cont='time', ax=ax)
ax = fig.add_subplot(grid[1, 1])
a.plot_continuation('PAR(14)', 'U(1)', cont='po', ax=ax)

# bifurcation diagram
ax = fig.add_subplot(grid[:, 0])
a.plot_continuation('PAR(4)', 'U(1)', cont='eta', ax=ax)
a.plot_continuation('PAR(4)', 'U(1)', cont='eta_hopf', ax=ax)

# finishing touches
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)
fig.canvas.draw()
fig.savefig('qif_sfa.pdf')
plt.show()
