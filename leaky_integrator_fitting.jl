using Distributed

addprocs(12)

@everywhere using BlackBoxOptim, LinearAlgebra, DifferentialEquations, NPZ, Plots

@everywhere function interp(x_new, x, y)
    idx = argmin(abs.(x.-x_new))
    if x[idx] > x_new
        i1, i2 = max(idx-1, 1), idx
    else
        i1, i2 = idx, min(idx+1, length(y))
    end
    return (y[i1] + y[i2])*0.5
end


@everywhere function li_eval(t,y,dy,r0,tau,source_idx,k_d1,k_d2,k_d3,k_d4,k_d5,k_d6,k_d7,k_d8,k_d9,k_d10,k_d11,u_timed_input,u_input,time,weight,weight_0)

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


@everywhere function l2_loss(x, y)
	diff = x .- y
	return sqrt(sum(sum(diff.^2)))
end

# import function arguments
@everywhere vars = npzread("li_params.npz")
@everywhere args = "r0,tau,source_idx,k_d1,k_d2,k_d3,k_d4,k_d5,k_d6,k_d7,k_d8,k_d9,k_d10,k_d11,u_timed_input,u_input,time,weight,weight_0"
@everywhere args = split(args, ",")

# basic problem parameters
@everywhere T = 100.0
@everywhere steps = 10000
@everywhere N = 5

# define functions for the parameter update
@everywhere idx_r = range(1, N)

@everywhere function update_connectivity(c)
	for i=1:N
		idx_c = range((i-1)*N+1,i*N)
		vars["weight"][idx_r, idx_c] = Diagonal(c[idx_c])
	end
end

@everywhere function ode_call(du, u, c, t)
	update_connectivity(c)
	return li_eval(t, u, du, [vars[key] for key in args]...)
end

# define the ODE problem
@everywhere w = zeros(N^2)
@everywhere ode = ODEProblem(ode_call, vars["y"], (0.0, T), w)

# define function call for blackboxoptim
@everywhere target = npzread("li_target.npy")
@everywhere z = target'
@everywhere solver = Tsit5()
@everywhere function optim(p)
	y = Array(solve(remake(ode, p=p), solver, saveat=1e-2, reltol=1e-3, abstol=1e-6))
	return l2_loss(y[1:N, 1:steps], z)
end

# display original connectivity matrix and target signal
@everywhere p1 = plot(target)
@everywhere p2 = heatmap(vars["weight"])

# define callback function for intermediate plotting
@everywhere function cb(oc)
	p = best_candidate(oc)
	y = Array(solve(remake(ode, p=p), solver, saveat=1e-2, reltol=1e-3, abstol=1e-6))
	p3 = plot(y[1:N, 1:steps]')
	p4 = heatmap(vars["weight"])
	display(plot(p3, p4, p1, p2, layout=(4,1)))
end

# perform optimization
method = :xnes
opt = bbsetup(optim; Method=method, Parameters=w, SearchRange=(-1.0, 1.0), NumDimensions=length(w), Workers=workers(),
	MaxSteps=1000, TargetFitness=0.0, lambda=10, PopulationSize=2000, CallbackFunction=cb, CallbackInterval=1.0)
el = @elapsed res = bboptimize(opt)

# retrieve optimization results
w_winner = best_candidate(res)
C = zeros(size(vars["weight"]))
for i=1:N
	idx_c = range((i-1)*N+1,i*N)
	C[idx_r, idx_c] = Diagonal(w_winner[idx_c])
end
f = best_fitness(res)

# simulate signal of the winner
y = Array(solve(remake(ode, p=w_winner), solver, saveat=1e-2, reltol=1e-3, abstol=1e-6))[1:N, 1:steps]

# save data to file
npzwrite("li_fitted.npz", Dict("weight" => C, "y" => y'))
