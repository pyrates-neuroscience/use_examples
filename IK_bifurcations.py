from pycobi import ODESystem
import numpy as np
import matplotlib.pyplot as plt
from pyrates import CircuitTemplate

# initialize mean-field model
ik_mf = CircuitTemplate.from_yaml("model_templates.neural_mass_models.ik.ik_theta")

# set the initial value of all model parameters
mf_params = {"C": 100.0, "k": 0.7, "v_r": -60.0, "v_t": -40.0, "b": -2.0, "g": 15.0,
             "a": 0.03, "d": 10.0, "tau_s": 6.0, "E_r": 0.0, "eta": 60.0, "Delta": 1.0}
ik_mf.update_var(node_vars={f"p/ik_theta_op/{key}": val for key, val in mf_params.items()})

# define simulation parameters (time unit: ms)
T = 2000.0
dt = 1e-2
sr = 100

# perform mean-field simulation
res = ik_mf.run(simulation_time=T, step_size=dt, sampling_step_size=int(dt*sr),
                outputs={"v": "p/ik_theta_op/v", "r": "p/ik_theta_op/r"},
                in_place=False, solver="scipy", method="RK45", atol=1e-7, rtol=1e-7)

# generate fortran files
ik_mf.get_run_func(func_name='ik_rhs', file_name='ik', step_size=dt, auto=True,
                   backend='fortran', solver='scipy', vectorize=False,
                   float_precision='float64')

# initialize pycobi
cont = ODESystem("ik", working_dir=None, auto_dir="~/PycharmProjects/auto-07p", init_cont=False)

# auto parameters
algorithm_params = {"NTST": 400, "NCOL": 4, "IAD": 3, "IPLT": 0, "NBC": 0, "NINT": 0, "NMX": 4000, "NPR": 10,
                    "MXBF": 5, "IID": 2, "ITMX": 40, "ITNW": 40, "NWTN": 12, "JAC": 0, "EPSL": 1e-7, "EPSU": 1e-7,
                    "EPSS": 1e-7, "DS": 1e-3, "DSMIN": 1e-8, "DSMAX": 5e-2, "IADS": 1, "THL": {}, "THU": {}}

# continuation of steady-state solution in eta (entry number 9 in the parameter vector)
eta_solutions, eta_cont = cont.run(name="eta:1", c="ivp", ICP=9, IPS=1, ILP=1, ISP=2, ISW=1, RL0=-20, RL1=80,
                                   UZR={9: 0.0}, STOP={}, bidirectional=True, **algorithm_params)

# plot the results
cont.plot_continuation("PAR(9)", "U(1)", cont="eta:1")
plt.show()

# continuation of user-defined point along steady state solution curve in SFA strength d (paramter number 17)
kappa_solutions, kappa_cont = cont.run(name="d", origin="eta:1", starting_point="UZ1", ICP=17, RL0=0, RL1=300,
                                       UZR={17: 100.0}, STOP={"UZ1"}, DS=1e-3)

# second continuation of steady-state solution in eta for large SFA strength
eta2_solutions, eta2_cont = cont.run(name="eta:2", origin="d", starting_point="UZ1", ICP=9, RL0=-20.0, RL1=100, UZR={},
                                     STOP={}, bidirectional=True)

# switch onto limit cycle branch and continue it in eta
lc_solutions, lc_cont = cont.run(name="eta:2:lc", origin="eta:2", starting_point="HB1", ICP=9, ISW=-1, IPS=2, ISP=2,
                                 STOP={"BP1", "LP2"}, RL0=-20.0, RL1=100.0, NMX=8000, DSMAX=0.1)

# plot the results
fig, ax = plt.subplots(figsize=(12, 6))
cont.plot_continuation("PAR(9)", "U(1)", cont="eta:2", ax=ax, line_color_stable="black")
cont.plot_continuation("PAR(9)", "U(1)", cont="eta:2:lc", ax=ax, line_color_stable='orange')
plt.show()

# perform a 2-parameter continuation of the fold and hopf curves in the eta-d parameter space
cont.run(name="eta_d_fold1", origin="eta:1", starting_point="LP1", ICP=[17, 9], ILP=0, IPS=1, ISW=2, NPR=50, RL0=0.0,
         RL1=200, bidirectional=True)
cont.run(name="eta_d_fold2", origin="eta:1", starting_point="LP2", ICP=[17, 9], bidirectional=True)
cont.run(name="eta_d_hopf1", origin="eta:2", starting_point="HB1", ICP=[17, 9], bidirectional=True)
cont.run(name="eta_d_hopf2", origin="eta:2", starting_point="HB2", ICP=[17, 9], bidirectional=True)

# plot the results
fig, ax = plt.subplots(figsize=(12, 6))
cont.plot_continuation("PAR(9)", "PAR(17)", cont="eta_d_fold1", ax=ax, line_color_stable="black")
cont.plot_continuation("PAR(9)", "PAR(17)", cont="eta_d_fold2", ax=ax, line_color_stable="black")
cont.plot_continuation("PAR(9)", "PAR(17)", cont="eta_d_hopf1", ax=ax, line_color_stable='orange')
cont.plot_continuation("PAR(9)", "PAR(17)", cont="eta_d_hopf2", ax=ax, line_color_stable='orange')
plt.show()