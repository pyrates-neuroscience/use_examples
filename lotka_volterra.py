import numpy as np
import matplotlib.pyplot as plt

# define lotka-volterra operator
################################

from pyrates import OperatorTemplate

op = OperatorTemplate(
    name="lv",
    equations="x' = x*(alpha + x_in)",
    variables={"x": "output(0.5)", "alpha": 0.5,
               "x_in": "input(0.0)"}
)

# define predator-prey model
############################

from pyrates import NodeTemplate, CircuitTemplate

# set up nodes for a predator and a prey population
alphas = {"predator": -1.0, "prey": 0.7}
nodes = {key: NodeTemplate(name="population", operators={op: {"alpha": alphas[key]}}) for key in ["predator", "prey"]}

# connect predator and prey population in a circuit
model = CircuitTemplate(
    name="network",
    nodes=nodes,
    edges=[
        ("predator/lv/x", "prey/lv/x_in", None, {"weight": -1.2, "delay": 0.01}),
        ("prey/lv/x", "predator/lv/x_in", None, {"weight": 1.0})
           ]
      )

# generate run function
#######################

# define simulation parameters
T = 100.0
dt = 1e-3

# generate function
func, args, _, _ = model.get_run_func(simulation_time=T, step_size=dt, func_name="lotka_volterra", solver="scipy",
                                      backend="numpy", in_place=False)

# perform simulation of DDE system via ddeint
#############################################

from ddeint import ddeint

# define ddeint wrapper function
t, y, hist, *fargs = args

def dde_run(y, t):
    return func(t, y(t), y, *fargs)

# solve DDE via ddeint
eval_time = np.linspace(0, T, num=int(T/dt))
res2 = ddeint(func=dde_run, g=hist, tt=eval_time)

# plot results
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(eval_time, res2[:, 0], label="predator")
ax.plot(eval_time, res2[:, 1], label="prey")
ax.set_xlabel('time')
ax.set_ylabel('x')
ax.legend()
plt.show()
