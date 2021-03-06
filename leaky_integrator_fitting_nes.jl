using LinearAlgebra, DifferentialEquations, NPZ, Plots, NaturalES

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
args = "r0,tau,source_idx,k_d1,k_d2,k_d3,k_d4,k_d5,k_d6,k_d7,k_d8,k_d9,k_d10,k_d11,u_timed_input,u_input,time,weight,weight_0"
args = split(args, ",")

# basic problem parameters
T = 100.0
steps = 10000
N = 5

# define functions for the parameter update
idx_r = range(1, N)

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

# define the ODE problem
w = zeros(N^2)
ode = ODEProblem(ode_call, vars["y"], (0.0, T), w)

# define function call for blackboxoptim
target = npzread("li_target.npy")
z = target'
p1 = plot(target)
p2 = heatmap(vars["weight"])
solver = Tsit5()
function objective_func(p)
	y = Array(DifferentialEquations.solve(remake(ode, p=p), solver, saveat=1e-2, reltol=1e-3, abstol=1e-6))
	p3 = plot(y[1:N, 1:steps]')
	p4 = heatmap(vars["weight"])
	display(plot(p3, p4, p1, p2, layout=(4,1)))
	return l2_loss(y[1:N, 1:steps], z)
end

# perform optimization
w_winner, fitness = optimize(objective_func, w, 1.0, sNES, iterations=1000, samples=100)

# retrieve optimization results
C = zeros(size(vars["weight"]))
for i=1:N
	idx_c = range((i-1)*N+1,i*N)
	C[idx_r, idx_c] = Diagonal(w_winner[idx_c])
end

# simulate signal of the winner
y = Array(DifferentialEquations.solve(remake(ode, p=w_winner), solver, saveat=1e-2, reltol=1e-3, abstol=1e-6))[1:N, 1:steps]

# save data to file
npzwrite("li_fitted.npz", Dict("weight" => C, "y" => y', "fitness" => fitness))
