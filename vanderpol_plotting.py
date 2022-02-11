from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
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

from scipy.signal import hilbert, butter, sosfilt, coherence


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
ax = fig.add_subplot(grid[:, :2])
im = ax.imshow(coherences[::-1, :], aspect='equal')
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$J$')
ax.set_xticks(np.arange(0, n_om, 3))
ax.set_yticks(np.arange(0, n_J, 3))
ax.set_xticklabels(np.round(omegas[::3], decimals=2))
ax.set_yticklabels(np.round(weights[::-3], decimals=2))
plt.title("Coherence between VPO and KO")
fig.colorbar(im, ax=ax, shrink=0.8)

# plot two exemplary time series
start = 900.0
for i, (omega, J) in enumerate(zip([0.32, 0.42], [0.5, 1.0])):

    ax = fig.add_subplot(grid[i, 2:])

    # find coherence matrix indices that corresponds to these parameters
    idx_r = np.argmin(np.abs(weights - J))
    idx_c = np.argmin(np.abs(omegas - omega))

    # find time series with that particular set of parameters
    idx1 = (res_map['VPO'].loc["J", :] == weights[idx_r]) * (res_map['VPO'].loc["omega", :] == omegas[idx_c])
    idx2 = (res_map['KO'].loc["J", :] == weights[idx_r]) * (res_map['KO'].loc["omega", :] == omegas[idx_c])
    key1, key2 = res_map['VPO'].columns.values[idx1], res_map['KO'].columns.values[idx2]

    # plot the time series
    ax.plot(res['VPO'].loc[start:, key1])
    ax.plot(res['KO'].loc[start:, key2])
    plt.legend(['Van der Pol', 'Kuramoto'])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Signal magnitude')

# finishing touches
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, hspace=0., wspace=0.)
fig.canvas.draw()
fig.savefig('vanderpol.pdf')
plt.show()
