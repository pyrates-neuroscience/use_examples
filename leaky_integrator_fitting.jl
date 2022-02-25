using Distributed

addprocs(8)

@everywhere using BlackBoxOptim, LinearAlgebra, DifferentialEquations, NPZ

@everywhere function interp(x_new, x, y)
    idx = argmin(abs.(x.-x_new))
    if x[idx] > x_new
        i1, i2 = max(idx-1, 1), idx
    else
        i1, i2 = idx, min(idx+1, length(y))
    end
    return (y[i1] + y[i2])*0.5
end


@everywhere function li_eval(t,y,dy,r0,tau,source_idx_0,k_d1,k_d2,k_d3,k_d4,k_d5,k_d6,k_d7,k_d8,k_d9,k_d10,k_d11,u_timed_input,target_idx,u_input,time,weight,source_idx,u)

	r = y[1:10]
	m_d1 = y[11:110]
	m_d2 = y[111:210]
	m_d3 = y[211:310]
	m_d4 = y[311:410]
	m_d5 = y[411:510]
	m_d6 = y[511:610]
	m_d7 = y[611:710]
	m_d8 = y[711:810]
	m_d9 = y[811:910]
	m_d10 = y[911:1010]
	m_d11 = y[1011:1110]

	m = @. tanh(r)
	m_buffered = @. m_d11
	u_timed_input[1] = interp(t, time, u_input)
	m_in = *(weight, m_buffered)
	u[target_idx] = @. u_timed_input[source_idx]

	dy[1:10] = @. m_in + u + (-r + r0)/tau
	dy[11:110] = @. k_d1*(-m_d1 + m[source_idx_0])
	dy[111:210] = @. k_d2*(m_d1 - m_d2)
	dy[211:310] = @. k_d3*(m_d2 - m_d3)
	dy[311:410] = @. k_d4*(m_d3 - m_d4)
	dy[411:510] = @. k_d5*(m_d4 - m_d5)
	dy[511:610] = @. k_d6*(m_d5 - m_d6)
	dy[611:710] = @. k_d7*(m_d6 - m_d7)
	dy[711:810] = @. k_d8*(m_d7 - m_d8)
	dy[811:910] = @. k_d9*(m_d8 - m_d9)
	dy[911:1010] = @. k_d10*(-m_d10 + m_d9)
	dy[1011:1110] = @. k_d11*(m_d10 - m_d11)

	return dy
end


@everywhere function l2_loss(x, y)
	diff = x .- y
	return sqrt(sum(sum(diff.^2)))
end

# import function arguments
@everywhere vars = npzread("/Users/rgf3807/PycharmProjects/use_examples/li_params.npz")
@everywhere args = "r0,tau,source_idx_0,k_d1,k_d2,k_d3,k_d4,k_d5,k_d6,k_d7,k_d8,k_d9,k_d10,k_d11,u_timed_input,target_idx,u_input,time,weight,source_idx,u"
@everywhere args = split(args, ",")

# basic problem parameters
@everywhere T = 100.0
@everywhere steps = 10000
@everywhere N = 10

# define function for the parameter update
@everywhere idx_r = range(1, N)
@everywhere function ode_call(du, u, c, t)
	for i=1:N
		idx_c = range((i-1)*N+1,i*N)
		vars["weight"][idx_r, idx_c] = Diagonal(c[idx_c])
	end
	return li_eval(t, u, du, [vars[key] for key in args]...)
end

@everywhere w = zeros(N^2)
@everywhere ode = ODEProblem(ode_call, vars["y"], (0.0, T), w)

# define function call for blackboxoptim
@everywhere target = npzread("/Users/rgf3807/PycharmProjects/use_examples/li_target.npy")
@everywhere z = target'
@everywhere function optim(p)
	y = Array(solve(remake(ode, p=p), Tsit5(), saveat=1e-2, reltol=1e-6, abstol=1e-6))
	return l2_loss(y[1:N, 1:steps], z)
end

# perform optimization
method = :separable_nes
opt = bbsetup(optim; Method=method, Parameters=w, SearchRange=(-10.1, 10.1), NumDimensions=length(w),
	MaxSteps=500, TargetFitness=0.0, PopulationSize=1000, lambda=10, Workers = workers())
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
y = Array(solve(remake(ode, p=w_winner), Tsit5(), saveat=1e-2, reltol=1e-6, abstol=1e-6))[1:N, 1:steps]

# save data to file
npzwrite("/Users/rgf3807/PycharmProjects/use_examples/li_fitted.npz", Dict("weight" => C, "y" => y'))
