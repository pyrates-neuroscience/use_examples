from matplotlib import gridspec
import matplotlib.pyplot as plt
from pyauto import PyAuto

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
a = PyAuto.from_file("qif_sfa_data.pkl", auto_dir='/home/rgast/PycharmProjects/auto-07p')

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

# time continuation
ax = fig.add_subplot(grid[0, 1])
ax = a.plot_continuation('PAR(14)', 'U(1)', cont='time', line_color_stable='k', ax=ax)
ax.set_xlabel('')
ax.set_ylabel('')
ax = fig.add_subplot(grid[1, 1])
ax = a.plot_continuation('PAR(14)', 'U(1)', cont='po', line_color_stable='orange', ax=ax)
ax.set_xlabel('time')
ax.set_ylabel(r'$r$')

# bifurcation diagram
ax = fig.add_subplot(grid[:, 0])
a.plot_continuation('PAR(4)', 'U(1)', cont='eta', ax=ax, ignore=['UZ', 'BP'])
ax = a.plot_continuation('PAR(4)', 'U(1)', cont='eta_hopf', ax=ax, line_color_stable='orange',
                         line_color_unstable='grey', ignore=['UZ', 'BP'])
ax.set_xlabel(r'$\bar\eta$')
ax.set_ylabel(r'$r$')

# finishing touches
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)
fig.canvas.draw()
fig.savefig('qif_sfa.pdf')
plt.show()
