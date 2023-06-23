from pycobi import ODESystem
import numpy as np
import matplotlib.pyplot as plt
from pyrates import CircuitTemplate

# initialize mean-field model
ik_mf = CircuitTemplate.from_yaml("model_templates.neural_mass_models.ik.ik_theta")

# set the initial value of all model parameters
mf_params = {"C": 100.0, "k": 0.7, "v_r": -60.0, "v_t": -40.0, "b": -2.0, "g": 15.0,
             "a": 0.03, "d": 10.0, "tau_s": 6.0, "E_r": 0.0, "eta": 60.0}
ik_mf.update_var(node_vars={f"p/ik_theta_op/{key}": val for key, val in mf_params.items()})

# define simulation parameters (time unit: ms)
T = 2000.0
dt = 1e-2
sr = 100

# perform mean-field simulation
res = ik_mf.run(simulation_time=T, step_size=dt, sampling_step_size=int(dt*sr),
                outputs={"v": "p/ik_theta_op/v", "r": "p/ik_theta_op/r"},
                in_place=False)

# generate fortran files
ik_mf.get_run_func(func_name='ik_rhs', file_name='ik', step_size=dt, auto=True,
                   backend='fortran', solver='scipy', vectorize=False,
                   float_precision='float64')

# initialize pycobi
cont = ODESystem(working_dir=None, auto_dir="~/PycharmProjects/auto-07p", init_cont=False)

# perform numerical integration in time
t_sols, t_cont = cont.run(
    e='ik', c='ivp', name='time', DS=1e-4, DSMIN=1e-10, EPSL=1e-08, NPR=10,
    EPSU=1e-08, EPSS=1e-06, DSMAX=1e-1, NMX=20000, UZR={14: 200.0}, STOP={'UZ1'}
    )

# auto parameters
algorithm_params = {"NTST": 400, "NCOL": 4, "IAD": 3, "IPLT": 0, "NBC": 0, "NINT": 0, "NMX": 2000, "NPR": 10,
                    "MXBF": 5, "IID": 2, "ITMX": 40, "ITNW": 40, "NWTN": 12, "JAC": 0, "EPSL": 1e-7, "EPSU": 1e-7,
                    "EPSS": 1e-7, "DS": 1e-3, "DSMIN": 1e-8, "DSMAX": 5e-2, "IADS": 1, "THL": {}, "THU": {}}

# continuation of steady-state solution in eta (entry number 9 in the parameter vector)
eta_solutions, eta_cont = cont.run(origin="time", starting_point="UZ1", name="eta",
                                   ICP=9, IPS=1, ILP=1, ISP=2, ISW=1, RL0=-50, RL1=150,
                                   UZR={}, STOP={}, bidirectional=True, **algorithm_params)

# plot the results
cont.plot_continuation("PAR(9)", "U(1)", cont="eta")
plt.show()
