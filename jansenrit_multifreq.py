"""
This script replicates the finding by David and Friston 2004 NeuroImage that the frequency of the Jansen-Rit model
can be tuned between delta and gamma frequencies by adapting the synaptic time constants.
"""

from pyrates import grid_search
import matplotlib.pyplot as plt
from seaborn import heatmap
from scipy.signal import welch
import numpy as np
from pandas import DataFrame

# define original synaptic efficacies and time constants
H_e_0 = 3.25e-3
H_i_0 = 22e-3
tau_e_0 = 10e-3
tau_i_0 = 20e-3

# define parameter grid to sweep over
n = 20
tau_e = np.linspace(1, 60, num=n)*1e-3
tau_i = np.linspace(1, 60, num=n)*1e-3
param_grid = {"tau_e": [], "tau_i": [], "h_e": [], "h_i": []}
for te in tau_e:
    for ti in tau_i:
        he = H_e_0*tau_e_0/te
        hi = H_i_0*tau_i_0/ti
        for key, val in zip(list(param_grid.keys()), [te, ti, he, hi]):
            param_grid[key].append(val)

# define mapping between grid entries and model parameters
param_map = {
    "tau_e": {"vars": ["jrc_op/tau_e"], "nodes": ["jrc"]},
    "tau_i": {"vars": ["jrc_op/tau_i"], "nodes": ["jrc"]},
    "h_e": {"vars": ["jrc_op/h_e"], "nodes": ["jrc"]},
    "h_i": {"vars": ["jrc_op/h_i"], "nodes": ["jrc"]}
             }

# define simulation parameters
T = 65.0
dt = 1e-3
cutoff = 5.0
solver_method = "RK45"
rtol = 1e-6
atol = 1e-6

# perform parameter sweep
res, res_map = grid_search("model_templates.neural_mass_models.jansenrit.JRC2",
                           outputs={"v_e": "jrc/jrc_op/V_e", "v_i": "jrc/jrc_op/V_i"},
                           param_grid=param_grid, param_map=param_map, simulation_time=T, step_size=dt,
                           solver="scipy", method=solver_method, rtol=rtol, atol=atol, cutoff=cutoff,
                           permute_grid=False)

# calculate pyramidal cell soma potential
v_pc = DataFrame(data=res["v_e"].values - res["v_i"].values, index=res.index, columns=res["v_e"].columns)

# get dominant oscillation frequency of JRC for each parameter combination
frequencies = np.zeros((n, n))
for key in res_map.index:
    signal = np.squeeze(v_pc[key].values)
    fs, ps = welch(signal, 1/dt, nfft=2048)
    max_freq = fs[np.argmax(ps)]
    col = np.argmin(np.abs(tau_e - res_map.at[key, "tau_e"]))
    row = np.argmin(np.abs(tau_i - res_map.at[key, "tau_i"]))
    frequencies[row, col] = max_freq

# plotting
fig, ax = plt.subplots(figsize=(8, 8))
heatmap(frequencies, cmap="Spectral", ax=ax, xticklabels=np.round(tau_e*1e3, decimals=1),
        yticklabels=np.round(tau_i*1e3, decimals=1))
ax.set_xlabel("tau_e (ms)")
ax.set_ylabel("tau_i (ms)")
ax.set_title("Maximum frequency (Hz) of the Jansen-Rit model")
ax.invert_yaxis()
plt.show()
