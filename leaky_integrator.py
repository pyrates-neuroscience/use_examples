import numpy as np
from pyrates import CircuitTemplate, NodeTemplate, clear_frontend_caches
clear_frontend_caches()

# node definition
li = NodeTemplate.from_yaml("model_templates.base_templates.tanh_node")
N = 3
nodes = [f"p{i+1}" for i in range(N)]
net = CircuitTemplate(name="li_coupled", nodes={key: li for key in nodes})

# edge definition
C = np.load("C.npy")  #np.random.uniform(low=-2.0, high=2.0, size=(N, N))
D = np.load("D.npy")  #np.random.choice([1.0, 2.0, 3.0], size=(N, N))
#np.save("C.npy", C)
#np.save("D.npy", D)
S = D*0.3
net.add_edges_from_matrix(source_var="tanh_op/m", target_var="li_op/m_in",
                          nodes=nodes, weight=C,
                          edge_attr={'delay': D, 'spread': S}
                          )

# define input
T = 150.0
dt = 1e-4
inp = np.sin(2 * np.pi * 0.2 * np.linspace(0, T, int(np.round(T / dt)))) * 0.1

# simulate time series
res = net.run(simulation_time=T, step_size=dt, sampling_step_size=1e-2,
              solver="scipy", method="RK45", outputs=["all/li_op/r"], rtol=1e-3, atol=1e-6,
              inputs={"all/li_op/u": inp}, backend="julia", clear=False,
              func_name="li_eval", constants_file_name="li_params.npz",
              julia_path="/Program Files/Julia/Julia-1.7.1/bin/julia.exe"
              )

import matplotlib.pyplot as plt
plt.plot(res)
plt.show()

# save results to file
np.save("li_target.npy", res)
