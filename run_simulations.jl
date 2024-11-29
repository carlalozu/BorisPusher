"""Plots the errors in x against different epsilon = 1/2^j are displayed"""
# To run script open julia terminal and run the following commands:
# import Pkg
# Pkg.activate("BorisPusher/Boris")
# include("BorisPusher/run_simulations.jl")

using Plots
using DifferentialEquations
using LinearAlgebra
using NLsolve

include("matrix_funcs.jl")
include("extras.jl")
include("utils.jl")
include("integrators.jl")

# initial position
x_0 = [1 / 3, 1 / 4, 1 / 2];
# initial velocity
v_0 = [2 / 5, 2 / 3, 1];

# numerical parameters, time
t0 = 0;
tf = 1;

# array to store the errors
errors_SB = Array{Float64}(undef, 13 - 4 + 1, 2);
errors_BEA = Array{Float64}(undef, 13 - 4 + 1, 2);
errors_BIA = Array{Float64}(undef, 13 - 4 + 1, 2);
errors_BT = Array{Float64}(undef, 13 - 4 + 1, 2);
epsilons = [];
i = 1;
for j in 4:13
    epsilon = (1 / 2)^j
    push!(epsilons, epsilon)

    h = epsilon
    nt = Int((tf - t0) / h + 1)
    println("j = ", j, ", nt = ", nt)

    # Runge Kutta (ground truth)
    x_tRK, v_tRK = runge_kutta(x_0, v_0, (t0, tf), nt, epsilon)
    v_tRK_p = parallel_velocity.(eachcol(x_tRK), eachcol(v_tRK))
    v_tRK_p = transpose(mapreduce(permutedims, vcat, v_tRK_p))

    # Standard Boris
    x_tSB, v_tSB = boris2(x_0, v_0, (t0, tf), nt, epsilon)
    v_tSB_p = parallel_velocity.(eachcol(x_tSB), eachcol(v_tSB))
    v_tSB_p = transpose(mapreduce(permutedims, vcat, v_tSB_p))

    # Explicit Filtered Boris
    x_tBEA, v_tBEA = boris_expA2(x_0, v_0, (t0, tf), nt, epsilon)
    v_tBEA_p = parallel_velocity.(eachcol(x_tBEA), eachcol(v_tBEA))
    v_tBEA_p = transpose(mapreduce(permutedims, vcat, v_tBEA_p))

    # Implicit Filtered Boris
    x_tBIA, v_tBIA = boris_impA2(x_0, v_0, (t0, tf), nt, epsilon)
    v_tBIA_p = parallel_velocity.(eachcol(x_tBIA), eachcol(v_tBIA))
    v_tBIA_p = transpose(mapreduce(permutedims, vcat, v_tBIA_p))

    # Two point filtered Boris
    x_tBT, v_tBT = boris_twoPA2(x_0, v_0, (t0, tf), nt, epsilon)
    v_tBT_p = parallel_velocity.(eachcol(x_tBT), eachcol(v_tBT))
    v_tBT_p = transpose(mapreduce(permutedims, vcat, v_tBT_p))

    # Global error in position
    errors_SB[i, 1] = sum(abs.(x_tRK .- x_tSB), dims=1)[nt]
    errors_BEA[i, 1] = sum(abs.(x_tRK .- x_tBEA), dims=1)[nt]
    errors_BIA[i, 1] = sum(abs.(x_tRK .- x_tBIA), dims=1)[nt]
    errors_BT[i, 1] = sum(abs.(x_tRK .- x_tBT), dims=1)[nt]

    # Global error in parallel velocities
    errors_SB[i, 2] = sum(abs.(v_tRK_p .- v_tSB_p), dims=1)[nt]
    errors_BEA[i, 2] = sum(abs.(v_tRK_p .- v_tBEA_p), dims=1)[nt]
    errors_BIA[i, 2] = sum(abs.(v_tRK_p .- v_tBIA_p), dims=1)[nt]
    errors_BT[i, 2] = sum(abs.(v_tRK_p .- v_tBT_p), dims=1)[nt]

    i += 1

end

savefig("errors_x_h_epsilon.png")