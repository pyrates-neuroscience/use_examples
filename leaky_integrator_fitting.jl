using LinearAlgebra, BlackBoxOptim, DifferentialEquations, NPZ


function interp(x_new, x, y)
    idx = argmin(abs.(x.-x_new))
    if x[idx] > x_new
        i1, i2 = max(idx-1, 1), idx
    else
        i1, i2 = idx, min(idx+1, length(y))
    end
    return (y[i1] + y[i2])*0.5
end


function li_eval(t,y,dy,r0,tau,source_idx_0,k_d1,k_d2,k_d3,k_d4,k_d5,k_d6,k_d7,k_d8,k_d9,k_d10,k_d11,k_d12,k_d13,r_buffered,u_timed_input,target_idx,target_idx_0,r_buffered_idx,u_input,time,weight,m_in,source_idx,u)

	r = y[1:10]
	r_d1 = y[11:110]
	r_d2 = y[111:187]
	r_d3 = y[188:243]
	r_d4 = y[244:285]
	r_d5 = y[286:314]
	r_d6 = y[315:338]
	r_d7 = y[339:350]
	r_d8 = y[351:358]
	r_d9 = y[359:364]
	r_d10 = y[365:368]
	r_d11 = y[369:369]
	r_d12 = y[370:370]
	r_d13 = y[371:371]

	r_buffered[1:23] = @. r_d1[1:23]
	r_buffered[24:44] = @. r_d2[1:21]
	r_buffered[45:58] = @. r_d3[1:14]
	r_buffered[59:71] = @. r_d4[1:13]
	r_buffered[72:76] = @. r_d5[1:5]
	r_buffered[77:88] = @. r_d6[1:12]
	r_buffered[89:92] = @. r_d7[1:4]
	r_buffered[93:94] = @. r_d8[1:2]
	r_buffered[95:96] = @. r_d9[1:2]
	r_buffered[97:99] = @. r_d10[1:3]
	r_buffered[100] = @. r_d13[1]
	r_buffered = @. r_buffered[r_buffered_idx]
	u_timed_input[1] = interp(t, time, u_input)
	m_in[target_idx] = @. r_buffered*weight
	u[target_idx_0] = @. u_timed_input[source_idx]

	dy[1:10] = @. m_in + u + (-r + r0)/tau
	dy[11:110] = @. k_d1*(-r_d1 + r[source_idx_0])
	dy[111:187] = @. k_d2*(-r_d2 + r_d1[24:100])
	dy[188:243] = @. k_d3*(-r_d3 + r_d2[22:77])
	dy[244:285] = @. k_d4*(-r_d4 + r_d3[15:56])
	dy[286:314] = @. k_d5*(-r_d5 + r_d4[14:42])
	dy[315:338] = @. k_d6*(-r_d6 + r_d5[6:29])
	dy[339:350] = @. k_d7*(-r_d7 + r_d6[13:24])
	dy[351:358] = @. k_d8*(-r_d8 + r_d7[5:12])
	dy[359:364] = @. k_d9*(-r_d9 + r_d8[3:8])
	dy[365:368] = @. k_d10*(-r_d10 + r_d9[3:6])
	dy[369:369] = @. k_d11*(-r_d11 + r_d10[4])
	dy[370:370] = @. k_d12*(r_d11 - r_d12)
	dy[371:371] = @. k_d13*(r_d12 - r_d13)

	return dy
end


function l2_loss(x, y)
	diff = x .- y
	return sqrt(sum(sum(diff.^2)))
end

# import function arguments
vars = npzread("/Users/rgf3807/PycharmProjects/use_examples/li_params.npz")
args = "r0,tau,source_idx_0,k_d1,k_d2,k_d3,k_d4,k_d5,k_d6,k_d7,k_d8,k_d9,k_d10,k_d11,k_d12,k_d13,r_buffered,u_timed_input,target_idx,target_idx_0,r_buffered_idx,u_input,time,weight,m_in,source_idx,u"
args = split(args, ",")

function ode_call(du, u, c, t)
	vars["weight"] = c
	return li_eval(t, u, du, [vars[key] for key in args]...)
end

# import target data
target = npzread("/Users/rgf3807/PycharmProjects/use_examples/li_target.npy")
z = target'
function optim(p)
	y = Array(solve(remake(ode, p=p), Tsit5(), saveat=1e-2, reltol=1e-6, abstol=1e-6))
	return l2_loss(y[1:10, 1:10000], z)
end

# setup ODE problem
T = 100.0
w = zeros(length(vars["weight"]))
ode = ODEProblem(ode_call, vars["y"], (0.0, T), w)

# define optimization parameter boundaries

# perform optimization
method = :dxnes
opt = bbsetup(optim; Method=method, Parameters=w, SearchRange=(-5.0, 5.0), NumDimensions=length(w), MaxSteps=50, TargetFitness=0.0, PopulationSize=1000)
el = @elapsed res = bboptimize(opt)

# retrieve optimization results
w_winner = best_candidate(res)
f = best_fitness(res)

# simulate signal of the winner
y = Array(solve(remake(ode, p=p), Tsit5(), saveat=1e-2, reltol=1e-6, abstol=1e-6))[1:10, 1:10000]

# print final result
show(sum(w .- w_winner))

# save data to file
npzwrite("/Users/rgf3807/PycharmProjects/use_examples/li_fitted.npz", Dict("weight" => w_winner, "y" => y'))
