using Distributed

addprocs(6)

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


@everywhere function li_eval(t,y,dy,r0,tau,source_idx_0,k_d1,k_d2,k_d3,k_d4,k_d5,k_d6,k_d7,k_d8,k_d9,r_buffered,u_timed_input,target_idx,target_idx_0,r_buffered_idx,u_input,time,weight,m_in,source_idx,u)

	r = y[1:10]
	r_d1 = y[11:110]
	r_d2 = y[111:210]
	r_d3 = y[211:310]
	r_d4 = y[311:410]
	r_d5 = y[411:510]
	r_d6 = y[511:610]
	r_d7 = y[611:648]
	r_d8 = y[649:686]
	r_d9 = y[687:724]

	r_buffered[1:62] = @. r_d6[1:62]
	r_buffered[63:100] = @. r_d9[1:38]
	r_buffered = @. r_buffered[r_buffered_idx]
	u_timed_input[1] = interp(t, time, u_input)
	m_in[target_idx] = @. r_buffered*weight
	u[target_idx_0] = @. u_timed_input[source_idx]

	dy[1:10] = @. u + tanh(m_in) + (-r + r0)/tau
	dy[11:110] = @. k_d1*(-r_d1 + r[source_idx_0])
	dy[111:210] = @. k_d2*(r_d1 - r_d2)
	dy[211:310] = @. k_d3*(r_d2 - r_d3)
	dy[311:410] = @. k_d4*(r_d3 - r_d4)
	dy[411:510] = @. k_d5*(r_d4 - r_d5)
	dy[511:610] = @. k_d6*(r_d5 - r_d6)
	dy[611:648] = @. k_d7*(-r_d7 + r_d6[63:100])
	dy[649:686] = @. k_d8*(r_d7 - r_d8)
	dy[687:724] = @. k_d9*(r_d8 - r_d9)

	return dy
end


@everywhere function l2_loss(x, y)
	diff = x .- y
	return sqrt(sum(sum(diff.^2)))
end

# import function arguments
@everywhere vars = npzread("/Users/rgf3807/PycharmProjects/use_examples/li_params.npz")
@everywhere args = "r0,tau,source_idx_0,k_d1,k_d2,k_d3,k_d4,k_d5,k_d6,k_d7,k_d8,k_d9,r_buffered,u_timed_input,target_idx,target_idx_0,r_buffered_idx,u_input,time,weight,m_in,source_idx,u"
@everywhere args = split(args, ",")

@everywhere function ode_call(du, u, c, t)
	vars["weight"] = c
	return li_eval(t, u, du, [vars[key] for key in args]...)
end

# import target data
@everywhere target = npzread("/Users/rgf3807/PycharmProjects/use_examples/li_target.npy")
@everywhere z = target'
@everywhere function optim(p)
	y = Array(solve(remake(ode, p=p), Tsit5(), saveat=1e-2, reltol=1e-6, abstol=1e-6))
	return l2_loss(y[1:10, 1:5000], z)
end

# setup ODE problem
@everywhere T = 50.0
@everywhere w = zeros(length(vars["weight"]))
@everywhere ode = ODEProblem(ode_call, vars["y"], (0.0, T), w)

# perform optimization
method = :separable_nes
opt = bbsetup(optim; Method=method, Parameters=w, SearchRange=(-5.0, 5.0), NumDimensions=length(w),
	MaxSteps=100, TargetFitness=0.0, PopulationSize=1000, lambda=100)
el = @elapsed res = bboptimize(opt)

# retrieve optimization results
w_winner = best_candidate(res)
f = best_fitness(res)

# simulate signal of the winner
y = Array(solve(remake(ode, p=w_winner), Tsit5(), saveat=1e-2, reltol=1e-6, abstol=1e-6))[1:10, 1:5000]

# print final result
show(sum(vars["weight"] .- w_winner))

# save data to file
npzwrite("/Users/rgf3807/PycharmProjects/use_examples/li_fitted.npz", Dict("weight" => w_winner, "y" => y'))
