from matplotlib import gridspec
import matplotlib.pyplot as plt
from pyauto import PyAuto

# preparations
##############

# plot settings
plt.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 400
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 14.0
plt.rcParams['axes.titlesize'] = 16.0
plt.rcParams['axes.labelsize'] = 16.0
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams["font.family"] = "sans-serif"
markersize = 200

# load data
a = PyAuto.from_file("qif_sfa_data.pkl", auto_dir='/home/rgast/PycharmProjects/auto-07p')

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

# time continuation
ax = fig.add_subplot(grid[0, 1])
a.plot_continuation('PAR(14)', 'U(1)', cont='ss', line_color_stable='k', ax=ax)
ax.set_xlim([-1.0, 70.0])
ax.set_xlabel('')
ax.set_ylabel(r'firing rate ($r$)')
plt.title(r'\textbf{(B)} Asynchronous regime')
ax = fig.add_subplot(grid[1, 1])
a.plot_continuation('PAR(14)', 'U(1)', cont='po', line_color_stable='orange', ax=ax)
ax.set_xlim([200.0, 350.0])
ax.set_xlabel(r'time (in units of $\tau$)')
ax.set_ylabel(r'firing rate ($r$)')
plt.title(r'\textbf{(C)} Synchronous regime')

# bifurcation diagram
ax = fig.add_subplot(grid[:, 0])
a.plot_continuation('PAR(4)', 'U(1)', cont='eta', ax=ax, ignore=['UZ', 'BP'], default_size=markersize)
a.plot_continuation('PAR(4)', 'U(1)', cont='eta_hopf', ax=ax, line_color_stable='orange',
                    line_color_unstable='grey', ignore=['UZ', 'BP'], default_size=markersize)
ax.set_xlabel(r'$\bar\eta$')
ax.set_ylabel(r'firing rate ($r$)')
ax.set_xlim([-10, 10])
plt.title(r'\textbf{(A)} 1D bifurcation diagram')

# add time series labels to the bifurcation diagram
for eta, y, label in zip([-2.0, 3.0], [3.3, 1.5], [r'\textbf{C}', r'\textbf{B}']):
    ax.text(eta, y, label, horizontalalignment='center', verticalalignment='center')

# finishing touches
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)
fig.canvas.draw()
fig.savefig('qif_sfa.pdf')
plt.show()
