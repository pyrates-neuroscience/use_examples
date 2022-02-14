import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, clear_frontend_caches
clear_frontend_caches()

# node definition
li = NodeTemplate.from_yaml("model_templates.base_templates.li_node")
N = 10
nodes = [f"p{i+1}" for i in range(N)]
net = CircuitTemplate(name="li_coupled", nodes={key: li for key in nodes})

# edge definition
C = np.random.uniform(low=-5.0, high=5.0, size=(N, N))
D = np.random.uniform(low=1.0, high=3.0, size=(N, N))
S = np.random.uniform(low=0.8, high=1.6, size=(N, N))
net.add_edges_from_matrix(source_var="li_op/r", target_var="li_op/m_in",
                          nodes=nodes, weight=C,
                          edge_attr={'delay': D, 'spread': S}
                          )

# define input
T = 100.0
dt = 1e-4
inp = np.sin(2 * np.pi * 0.2 * np.linspace(0, T, int(np.round(T / dt)))) * 0.1

# simulate time series
res = net.run(simulation_time=T, step_size=dt, sampling_step_size=1e-2,
              solver="scipy", method="DOP853", outputs={"all/li_op/r"},
              inputs={"all/li_op/u": inp}, backend="julia", clear=False,
              constants_file_name="li_params.npz", func_name="li_eval",
              julia_path="/Program Files/Julia/Julia-1.7.1/bin/julia.exe")

import matplotlib.pyplot as plt
plt.plot(res)
plt.show()

# save results to file
np.save("li_target.npy", res)
