import pandas as pd
from rectipy import Network
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle

# select device on which to run the optimization
device = "cpu"

# preparations
##############

# model parameters
node = "model_definitions/tanh"
N = 5
J = np.random.uniform(low=-1.0, high=1.0, size=(N, N))  # np.load("J.npy")
D = np.random.choice([1.0, 2.0, 3.0], size=(N, N))  #np.load("D.npy")
np.save("J.npy", J)
np.save("D.npy", D)
S = D*0.3
pmin, pmax = 0.1, 10.0
k0 = np.random.uniform(pmin, pmax)
tau0 = np.random.uniform(pmin, pmax)
v0 = np.random.randn(N)
dt = 1e-3

# initialize networks
target_net = Network(dt=dt, device=device)
target_net.add_diffeq_node("tanh", node=node, weights=J, edge_attr={'delay': D, 'spread': S}, source_var="tanh_op/r",
                           target_var="li_op/r_in", input_var="li_op/I_ext", output_var="li_op/v", clear=True,
                           float_precision="float64", file_name='target_net',
                           node_vars={'all/li_op/v': v0})
learning_net = Network(dt=dt, device=device)
learning_net.add_diffeq_node("tanh", node=node, weights=J, edge_attr={'delay': D, 'spread': S}, source_var="tanh_op/r",
                             target_var="li_op/r_in", input_var="li_op/I_ext", output_var="li_op/v", clear=True,
                             train_params=['li_op/k', 'li_op/tau'], float_precision="float64",
                             node_vars={"all/li_op/k": k0, "all/li_op/tau": tau0}, file_name='learning_net')

# compile networks
target_net.compile()
learning_net.compile()

# extract initial value vector for later state vector resets
y0 = {key: val.clone() for key, val in target_net.state.items()}

# create target data
####################

# error parameters
tol = 1e-3
error = 1.0

# input parameters
freq = 0.2
amp = 0.1

# epoch parameters
n_epochs = 100
epoch_steps = 30000
epoch = 0

# target data creation
print("Creating target data...")
target_net.reset(y0)
targets = []
for step in range(epoch_steps):
    inp = np.sin(2 * np.pi * freq * step * dt) * amp
    target = target_net.forward(inp)
    targets.append(target)
print("Finished.")

# map out loss landscape
########################

# loss function definition
loss = torch.nn.MSELoss()

# parameter space definition
n = 10
vals = 10.0**np.linspace(-1.0, 1.0, num=n)

# network initialization
net = Network(dt=dt, device=device)
net.add_diffeq_node("tanh", node=node, weights=J, edge_attr={'delay': D, 'spread': S}, source_var="tanh_op/r",
                    target_var="li_op/r_in", input_var="li_op/I_ext", output_var="li_op/v", clear=True,
                    verbose=False, float_precision="float64")
net.compile()

# loss landscape mapping
print("Approximating loss landscape...")
loss_2d = np.zeros((n, n))
for i, k in enumerate(vals):
    for j, tau in enumerate(vals):

        # reset the network state
        net.reset(y0)

        # change network parameter values
        for key, val in {'li_op/tau': float(tau), 'li_op/k': float(k)}.items():
            net.set_var("tanh", key, val)

        # collect the losses for each value of the 2D parameter grid
        losses = []
        for step in range(epoch_steps):
            inp = np.sin(2 * np.pi * freq * step * dt) * amp
            prediction = net.forward(inp)
            error_tmp = loss(prediction, targets[step])
            losses.append(error_tmp.item())
        loss_2d[i, j] = np.mean(losses)

    # report progress
    print(f"Progress: {np.round(100*(i+1)/n, decimals=0)} %")

# display loss landscape
plt.imshow(np.log(loss_2d))
plt.colorbar()
plt.show()

# turn loss landscape into a dataframe
loss_2d = pd.DataFrame(index=vals, columns=vals, data=loss_2d)

# optimization
##############

# optimizer definition
opt = torch.optim.Rprop(learning_net.parameters(), lr=0.01, etas=(0.5, 1.1), step_sizes=(1e-5, 1e-1))

# optimization loop
print("Starting optimization...")
losses, ks, taus = [], [], []
error_tmp = torch.zeros(1)
while error > tol and epoch < n_epochs:

    # error calculation epoch
    losses_tmp = []
    learning_net.reset(y0)
    for step in range(epoch_steps):
        inp = np.sin(2*np.pi*freq*step*dt) * amp
        target = targets[step]
        prediction = learning_net.forward(inp)
        error_tmp += loss(prediction, target)
        losses_tmp.append(error_tmp.item())

    # optimization step
    opt.zero_grad()
    error_tmp.backward()
    opt.step()
    error_tmp = torch.zeros(1)

    # save results and display progress
    error = np.mean(losses_tmp)
    losses.append(error)
    ks.append(learning_net.get_var("tanh", "li_op/k").clone().detach().cpu().numpy())
    taus.append(learning_net.get_var("tanh", "li_op/tau").clone().detach().cpu().numpy())
    epoch += 1
    print(f"Training epoch #{epoch} finished. Mean epoch loss: {error}.")

# model testing
###############

print("Starting testing...")
learning_net.reset(y0)
predictions = []
for step in range(epoch_steps):
    inp = np.sin(2 * np.pi * freq * step * dt) * amp
    prediction = learning_net.forward(inp)
    predictions.append(prediction.detach().cpu().numpy())
print("Finished.")

# saving data to file
#####################

targets = [t.detach().numpy() for t in targets]
target_vals = [target_net.get_var("tanh", "li_op/k").detach().cpu().numpy(),
               target_net.get_var("tanh", "li_op/tau").detach().cpu().numpy()]
orig_vals = [k0, tau0]
fitted_vals = [learning_net.get_var("tanh", "li_op/k").detach().cpu().numpy(),
               learning_net.get_var("tanh", "li_op/tau").detach().cpu().numpy()]
parameters = ["J", "tau"]
pickle.dump({"predictions": predictions, "targets": targets, "loss": losses, "ks": ks, "taus": taus,
             "params": parameters, "fitted": fitted_vals, "original": orig_vals, "true": target_vals,
             "loss_landscape": loss_2d},
            open("leaky_integrator_data.pkl", "wb"))

# plotting
##########

fig, axes = plt.subplots(nrows=3, figsize=(12, 8))
ax1 = axes[0]
ax1.plot(predictions)
ax1.set_title('predictions (testing)')
ax1.set_xlabel('steps')
ax1.set_ylabel('u')
ax2 = axes[1]
ax2.plot(targets)
ax2.set_title('targets (testing)')
ax2.set_xlabel('steps')
ax2.set_ylabel('u')
ax3 = axes[2]
ax3.plot(losses)
ax3.set_title('loss (training)')
ax3.set_xlabel('epochs')
ax3.set_ylabel('MSE')
plt.tight_layout()

for key, val, target, start in zip(parameters, fitted_vals, target_vals, orig_vals):
    print(f"Parameter: {key}. Target: {target}. Fitted value: {val}. Initial value: {start}.")
plt.show()
