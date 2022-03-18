using LinearAlgebra, DifferentialEquations, NPZ, Plots, GalacticOptim, BlackBoxOptim

gr()

function interp(x_new, x, y)
    idx = argmin(abs.(x.-x_new))
    if x[idx] > x_new
        i1, i2 = max(idx-1, 1), idx
    else
        i1, i2 = idx, min(idx+1, length(y))
    end
    return (y[i1] + y[i2])*0.5
end


function li_eval(t,y,dy,r0,tau,source_idx,k_d1,k_d2,k_d3,k_d4,k_d5,k_d6,k_d7,k_d8,k_d9,k_d10,k_d11,u_timed_input,u_input,time,weight,weight_0)

	r = y[1:5]
	m_d1 = y[6:30]
	m_d2 = y[31:55]
	m_d3 = y[56:80]
	m_d4 = y[81:105]
	m_d5 = y[106:130]
	m_d6 = y[131:155]
	m_d7 = y[156:180]
	m_d8 = y[181:205]
	m_d9 = y[206:230]
	m_d10 = y[231:255]
	m_d11 = y[256:280]

	m = @. tanh(r)
	m_buffered = @. m_d11
	u_timed_input[1] = interp(t, time, u_input)
	m_in = *(weight, m_buffered)
	u = *(weight_0, u_timed_input)

	dy[1:5] = @. m_in + u + (-r + r0)/tau
	dy[6:30] = @. k_d1*(-m_d1 + m[source_idx])
	dy[31:55] = @. k_d2*(m_d1 - m_d2)
	dy[56:80] = @. k_d3*(m_d2 - m_d3)
	dy[81:105] = @. k_d4*(m_d3 - m_d4)
	dy[106:130] = @. k_d5*(m_d4 - m_d5)
	dy[131:155] = @. k_d6*(m_d5 - m_d6)
	dy[156:180] = @. k_d7*(m_d6 - m_d7)
	dy[181:205] = @. k_d8*(m_d7 - m_d8)
	dy[206:230] = @. k_d9*(m_d8 - m_d9)
	dy[231:255] = @. k_d10*(-m_d10 + m_d9)
	dy[256:280] = @. k_d11*(m_d10 - m_d11)

	return dy
end


function l2_loss(x, y)
	diff = x .- y
	return sqrt(sum(sum(diff.^2)))
end

# import function arguments
vars = npzread("li_params.npz")
w = npzread("li_fitted_o.npz")["weight"]
args = "r0,tau,source_idx,k_d1,k_d2,k_d3,k_d4,k_d5,k_d6,k_d7,k_d8,k_d9,k_d10,k_d11,u_timed_input,u_input,time,weight,weight_0"
args = split(args, ",")

# basic problem parameters
T = 100.0
steps = 10000
N = 5
y_shape = size(vars["y"])

# extract pre-fitted connectivity
idx_r = range(1, N)
n_par = N^2
w_0 = zeros(n_par)
for i=1:N
	idx_c = range((i-1)*N+1,i*N)
	w_tmp = Diagonal(w[idx_r, idx_c])
	w_0[idx_c] = Array(w_tmp[abs.(w_tmp) .> 0.0])
end

# define functions for the parameter update
function update_connectivity(c)
	c_new = deepcopy(vars["weight"])
	for i=1:N
		idx_c = range((i-1)*N+1,i*N)
		c_new[idx_r, idx_c] = Diagonal(c[idx_c])
	end
	return c_new
end

function ode_call(du, u, c, t)
	c_new = update_connectivity(c)
	return li_eval(t, u, du, [key == "weight" ? c_new : deepcopy(vars[key]) for key in args]...)
end

# define function call for blackboxoptim
target = npzread("li_target.npy")
z = target'
solver = Tsit5()
function objective_func(p, p_tmp)
	ode = ODEProblem(ode_call, zeros(y_shape), (0.0, T), p)
	y = Array(DifferentialEquations.solve(ode, solver, saveat=1e-2, reltol=1e-3, abstol=1e-6))
	return l2_loss(y[1:N, 1:steps], z)
end

# display original connectivity matrix and target signal
p1 = plot(target)
p2 = heatmap(vars["weight"])

# define callback function for intermediate plotting
t0 = time()
function cb(p, p_tmp)
	t1 = time()
	ode = ODEProblem(ode_call, zeros(y_shape), (0.0, T), p)
	y = Array(DifferentialEquations.solve(ode, solver, saveat=1e-2, reltol=1e-3, abstol=1e-6))
	p3 = plot(y[1:N, 1:steps]')
	c_new = update_connectivity(p)
	p4 = heatmap(c_new)
	display(plot(p3, p4, p1, p2, layout=(4,1)))
	print(t1-t0)
	print("\n")
	return t1-t0 > 5000 ? true : false
end

# define optimization problem
f = OptimizationFunction(objective_func)
lower = [w_0[i] > 0 ? 0.0 : -1.0 for i=1:n_par]
upper = [w_0[i] > 0 ? 1.0 : 0.0 for i=1:n_par]
method = BBO_adaptive_de_rand_1_bin_radiuslimited()
f = GalacticOptim.OptimizationFunction(objective_func)
prob = GalacticOptim.OptimizationProblem(f, w_0, lb=lower, ub=upper)

# perform optimization
res = GalacticOptim.solve(prob, method, maxiters=1000, cb=cb)

# retrieve optimization results
w_winner = res.u
C = zeros(size(vars["weight"]))
for i=1:N
	idx_c = range((i-1)*N+1,i*N)
	C[idx_r, idx_c] = Diagonal(w_winner[idx_c])
end

# simulate signal of the winner
ode = ODEProblem(ode_call, zeros(y_shape), (0.0, T), w_winner)
y = Array(DifferentialEquations.solve(ode, solver, saveat=1e-2, reltol=1e-3, abstol=1e-6))[1:N, 1:steps]

# plot the final result
cb(w_winner, [])

# save data to file
npzwrite("li_fitted.npz", Dict("weight" => C, "y" => y', "fitness" => res.minimum))
