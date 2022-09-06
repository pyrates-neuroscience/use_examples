import numpy as np
from pyrates import CircuitTemplate
from pyauto import PyAuto

# PyRates initiation
qif = CircuitTemplate.from_yaml(
    "model_templates.neural_mass_models.qif.qif_sfa"
    )

# set model parameters
qif.update_var(node_vars={'p/qif_sfa_op/Delta': 2.0, 'p/qif_sfa_op/alpha': 1.0, 'p/qif_sfa_op/eta': -10.0},
               edge_vars=[('p/qif_sfa_op/r', 'p/qif_sfa_op/r_in', {'weight': 15.0*np.sqrt(2.0)})])

# run function generation
qif.get_run_func(func_name='qif_run', file_name='qif_sfa', step_size=1e-4,
                 backend='fortran', solver='scipy', vectorize=False,
                 float_precision='float64', auto=True)

# initialize PyAuto
qif_auto = PyAuto(working_dir=None, auto_dir='/home/rgast/PycharmProjects/auto-07p')

# perform numerical integration in time
t_sols, t_cont = qif_auto.run(
    e='qif_sfa', c='ivp', name='time', DS=1e-4, DSMIN=1e-10, EPSL=1e-08, NPR=100,
    EPSU=1e-08, EPSS=1e-06, DSMAX=1e-2, NMX=20000, UZR={14: 150.0}, STOP={'UZ1'}
    )

# continue eta
eta_sols, eta_cont = qif_auto.run(
    origin=t_cont, starting_point='UZ1', name='eta', bidirectional=True,
    ICP=4, RL0=-20.0, RL1=20.0, IPS=1, ILP=1, ISP=2, ISW=1, NTST=400,
    NCOL=4, IAD=3, IPLT=0, NBC=0, NINT=0, NMX=2000, NPR=10, MXBF=5, IID=2,
    ITMX=40, ITNW=40, NWTN=12, JAC=0, EPSL=1e-06, EPSU=1e-06, EPSS=1e-04,
    DS=1e-4, DSMIN=1e-8, DSMAX=5e-2, IADS=1, THL={}, THU={}, UZR={4: 3.0}, STOP={}
)

# continue periodic solution
hopf_sols, hopf_cont = qif_auto.run(
    origin=eta_cont, starting_point='HB2', name='eta_hopf',
    IPS=2, ISP=2, ISW=-1, UZR={4: -2.0}
)

# periodic solution time continuation
ss_sols, ss_cont = qif_auto.run(
    origin=eta_cont, starting_point='UZ1', name='ss', DS=1e-4, DSMIN=1e-10, EPSL=1e-08, NPR=10,
    EPSU=1e-08, EPSS=1e-06, DSMAX=1e-1, NMX=10000, UZR={14: 300.0}, STOP={'UZ1'}, c='ivp'
    )

# periodic solution time continuation
po_sols, po_cont = qif_auto.run(
    origin=hopf_cont, starting_point='UZ1', name='po', DS=1e-4, DSMIN=1e-10, EPSL=1e-08, NPR=10,
    EPSU=1e-08, EPSS=1e-06, DSMAX=1e-1, NMX=10000, UZR={14: 300.0}, STOP={'UZ1'}, c='ivp'
    )

qif_auto.to_file('qif_sfa_data.pkl')
