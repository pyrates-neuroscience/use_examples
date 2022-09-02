from rectipy import Network
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle


# preparations
##############

# model parameters
node = "neuron_model_templates.rate_neurons.leaky_integrator.tanh_pop"
N = 5
J = np.load("J.npy")  #np.random.uniform(low=-1.0, high=1.0, size=(N, N))
D = np.load("D.npy")  #np.random.choice([1.0, 2.0, 3.0], size=(N, N))
#np.save("J.npy", C)
#np.save("D.npy", D)
S = D*0.3
pmin, pmax = 0.1, 10.0
k0 = np.random.uniform(pmin, pmax)
tau0 = np.random.uniform(pmin, pmax)
u0 = np.random.randn(N)

# initialize networks
target_net = Network.from_yaml(node=node, weights=J, edge_attr={'delay': D, 'spread': S}, source_var="tanh_op/r",
                               target_var="li_op/r_in", input_var="li_op/I_ext", output_var="li_op/u", clear=True,
                               float_precision="float64", file_name='target_net',
                               node_vars={'all/li_op/u': u0})

learning_net = Network.from_yaml(node=node, weights=J, edge_attr={'delay': D, 'spread': S}, source_var="tanh_op/r",
                                 target_var="li_op/r_in", input_var="li_op/I_ext", output_var="li_op/u", clear=True,
                                 train_params=['li_op/k', 'li_op/tau'], float_precision="float64",
                                 node_vars={"all/li_op/k": k0, "all/li_op/tau": tau0}, file_name='learning_net')

# compile networks
target_net.compile()
learning_net.compile()

# extract initial value vector for later state vector resets
y0 = target_net.rnn_layer.y.clone().detach().numpy()

# create target data
####################

# error parameters
tol = 1e-3
error = 1.0

# input parameters
dt = 1e-3
freq = 0.2
amp = 0.1

# epoch parameters
n_epochs = 100
disp_steps = 1000
epoch_steps = 30000
epoch = 0

# target data creation
print("Creating target data...")
target_net.rnn_layer.reset(y0)
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
n = 50
vals = 10.0**np.linspace(-1.0, 1.0, num=n)

# loss landscape mapping
loss_2d = np.zeros((n, n))
for i, k in enumerate(vals):
    for j, tau in enumerate(vals):
        net = Network.from_yaml(node=node, weights=J, edge_attr={'delay': D, 'spread': S}, source_var="tanh_op/r",
                                target_var="li_op/r_in", input_var="li_op/I_ext", output_var="li_op/u", clear=True,
                                node_vars={'all/li_op/u': u0, 'all/li_op/tau': float(tau), 'all/li_op/k': float(k)},
                                verbose=False, float_precision="float64")
        net.compile()
        losses = []
        for step in range(epoch_steps):
            inp = np.sin(2 * np.pi * freq * step * dt) * amp
            prediction = net.forward(inp)
            error_tmp = loss(prediction, targets[step])
            losses.append(error_tmp.item())
        loss_2d[i, j] = np.mean(losses)

plt.imshow(np.log(loss_2d))
plt.colorbar()
plt.show()

# optimization
##############

# optimizer definition
opt = torch.optim.Rprop(learning_net.parameters(), lr=0.01, etas=(0.5, 1.1), step_sizes=(1e-5, 1e-1))

# optimization loop
losses, ks, taus = [], [], []
while error > tol and epoch < n_epochs:

    # error calculation epoch
    losses_tmp = []
    learning_net.rnn_layer.reset(y0)
    for step in range(epoch_steps):
        inp = np.sin(2*np.pi*freq*step*dt) * amp
        target = targets[step]
        prediction = learning_net.forward(inp)
        error_tmp = loss(prediction, target)
        error_tmp.backward(retain_graph=True)
        if step % disp_steps == 0:
            print(f"Steps finished: {step}. Current loss: {error_tmp.item()}")
        losses_tmp.append(error_tmp.item())

    # optimization step
    opt.step()
    opt.zero_grad()

    # save results and display progress
    error = np.mean(losses_tmp)
    losses.append(error)
    ks.append(learning_net.rnn_layer.args[1].clone().detach().numpy())
    taus.append(learning_net.rnn_layer.args[0].clone().detach().numpy())
    epoch += 1
    print(f"Training epoch #{epoch} finished. Mean epoch loss: {error}.")

# model testing
###############

print("Starting testing...")
learning_net.rnn_layer.reset(y0)
predictions = []
for step in range(epoch_steps):
    inp = np.sin(2 * np.pi * freq * step * dt) * amp
    prediction = learning_net.forward(inp)
    predictions.append(prediction.detach().numpy())
print("Finished.")

# saving data to file
#####################

targets = [t.detach().numpy() for t in targets]
target_vals = [target_net.rnn_layer.args[1].numpy(), target_net.rnn_layer.args[0].numpy()]
orig_vals = [k0, tau0]
fitted_vals = [learning_net.rnn_layer.args[1].detach().numpy(), learning_net.rnn_layer.args[0].detach().numpy()]
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
