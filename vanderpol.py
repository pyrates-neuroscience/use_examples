from pyrates import CircuitTemplate, NodeTemplate

# define nodes
VPO = NodeTemplate.from_yaml(
    "model_templates.coupled_oscillators.vanderpol.vdp_pop"
    )
KO = NodeTemplate.from_yaml(
    "model_templates.coupled_oscillators.kuramoto.sin_pop"
    )

# define network
net = CircuitTemplate(
    name="VPO_forced", nodes={'VPO': VPO, 'KO': KO},
    edges=[('KO/sin_op/s', 'VPO/vdp_op/inp', None, {'weight': 1.0})]
    )

# adjust the damping parameter of the VPO
net.update_var({'VPO/vdp_op/mu': 2.0})

# imports
import numpy as np
from pyrates import grid_search

# define parameter sweep
n_om = 20
n_J = 20
omegas = np.linspace(0.3, 0.5, num=n_om)
weights = np.linspace(0.0, 2.0, num=n_J)

# map sweep parameters to network parameters
params = {'omega': omegas, 'J': weights}
param_map = {'omega': {'vars': ['phase_op/omega'],
                       'nodes': ['KO']},
             'J': {'vars': ['weight'],
                   'edges': [('KO/sin_op/s', 'VPO/vdp_op/inp')]}
            }

# perform parameter sweep
T = 1100.0
dt = 1e-3
dts = 1e-2
cutoff = 100.0
results, res_map = grid_search(
    circuit_template=net, param_grid=params, param_map=param_map,
    simulation_time=T, step_size=dt, solver='scipy', method='DOP853',
    outputs={'VPO': 'VPO/vdp_op/x', 'KO': 'KO/phase_op/theta'},
    inputs=None, vectorize=True, clear=False, file_name='vpo_forced',
    permute_grid=True, cutoff=cutoff, sampling_step_size=dts
    )

# save results to file
import pickle
data = {'res': results, 'map': res_map, 'omegas': omegas, 'weights': weights}
fn = 'vanderpol_data.pkl'
try:
    pickle.dump(data, open(fn, 'x'))
except (FileExistsError, TypeError):
    pickle.dump(data, open(fn, 'wb'))
