from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.signal import hilbert, butter, sosfilt, coherence
import os
p = os.environ['PATH']
plt.rc('text', usetex=True)

# preparations
##############

# plot settings
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 400
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 14.0
plt.rcParams['axes.titlesize'] = 16.0
plt.rcParams['axes.labelsize'] = 16.0
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams["font.family"] = "sans-serif"

# load data
data = pickle.load(open('vanderpol_data.pkl', 'rb'))
res = data['res']
res_map = data['map']
omegas = data['omegas']
weights = data['weights']
n_om = len(omegas)
n_J = len(weights)

# coherence calculation parameters
nps = 1024
window = 'hamming'
fs = 1/(res.index[1] - res.index[0])

# calculate coherences
######################


def get_phase(signal, N, freqs, fs):
    """Extracts phase from signal using a butterworth bandpass filter.

    :param signal: 1D array.
    :param N: Order of the filter.
    :param freqs: Tuple with low- and highpass frequency cutoff.
    :param fs: Sampling frequency of the signal.
    :return: Instantaneous phase of the signal in the bandpass frequency band.
    """
    filt = butter(N, freqs, output="sos", btype='bandpass', fs=fs)
    s_filtered = sosfilt(filt, signal)
    return np.unwrap(np.angle(hilbert(s_filtered)))


# calculate and store coherences
coherences = np.zeros((n_J, n_om))
for key1, key2 in zip(res_map['VPO'].columns, res_map['KO'].columns):

    # extract parameter set
    omega = res_map['KO'].at['omega', key2]
    J = res_map['KO'].at['J', key2]

    # find coherence matrix position that corresponds to these parameters
    idx_r = np.argmin(np.abs(weights - J))
    idx_c = np.argmin(np.abs(omegas - omega))

    # collect phases
    tf = np.maximum(0.01, omegas[idx_c])
    p1 = np.sin(get_phase(res['VPO'].loc[:, key1].squeeze(), N=10,
                          freqs=(tf-0.3*tf, tf+0.3*tf), fs=fs))
    p2 = np.sin(2*np.pi*res['KO'].loc[:, key2].squeeze())

    # calculate coherence
    freq, coh = coherence(p1, p2, fs=fs, nperseg=nps, window=window)

    # store coherence value at driving frequency
    tf = freq[np.argmin(np.abs(freq - omega))]
    coherences[idx_r, idx_c] = np.max(coh[(freq >= tf-0.3*tf) * (freq <= tf+0.3*tf)])

# plotting
##########

# create figure layout
fig = plt.figure(1)
grid = gridspec.GridSpec(nrows=2, ncols=4, figure=fig)

# plot the coherence matrix
ax1 = fig.add_subplot(grid[:, :2])
im = ax1.imshow(coherences[::-1, :], aspect='equal')
ax1.set_xlabel(r'$\omega$')
ax1.set_ylabel(r'$J$')
ax1.set_xticks(np.arange(0, n_om, 3))
ax1.set_yticks(np.arange(0, n_J, 3))
ax1.set_xticklabels(np.round(omegas[::3], decimals=2))
ax1.set_yticklabels(np.round(weights[::-3], decimals=2))
plt.title(r"\textbf{(A)} Coherence between VPO and KO")
fig.colorbar(im, ax=ax1, shrink=0.6, label='signal (normalized amplitude)')

# plot two exemplary time series
start = 1050.0
for i, (omega, J, cond, title) in enumerate(zip([0.32, 0.42], [0.5, 1.0], [(r"\textbf{B}", "white"), (r"\textbf{C}", "black")],
                                                [r"\textbf{(B)} No entrainment", r"\textbf{(C)} Entrainment"])):

    ax = fig.add_subplot(grid[i, 2:])

    # find coherence matrix indices that corresponds to these parameters
    idx_r = np.argmin(np.abs(weights - J))
    idx_c = np.argmin(np.abs(omegas - omega))

    # find time series with that particular set of parameters
    idx1 = (res_map['VPO'].loc["J", :] == weights[idx_r]) * (res_map['VPO'].loc["omega", :] == omegas[idx_c])
    idx2 = (res_map['KO'].loc["J", :] == weights[idx_r]) * (res_map['KO'].loc["omega", :] == omegas[idx_c])
    key1, key2 = res_map['VPO'].columns.values[idx1], res_map['KO'].columns.values[idx2]

    # plot the time series
    s = res['VPO'].loc[start:, key1]
    ax.plot(s / np.max(np.abs(s)), color='black')
    ax.plot(np.sin(2*np.pi*res['KO'].loc[start:, key2]), color='orange')
    ax.set_title(title)

    # add figure label to the imshow plot
    ax1.text(idx_c, n_J-idx_r, cond[0], color=cond[1], horizontalalignment='center', verticalalignment='center')

# finishing touches
ax.set_xlabel('time (s)')
plt.legend(['Van der Pol', 'Kuramoto'], facecolor="gray", loc="upper right")
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)
fig.canvas.draw()
fig.savefig('vanderpol.pdf')
plt.show()
