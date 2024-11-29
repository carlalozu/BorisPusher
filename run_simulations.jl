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
errors_SB = Array{Float64}(undef, 13 - 4 + 1, 3);
errors_BEA = Array{Float64}(undef, 13 - 4 + 1, 3);
errors_BIA = Array{Float64}(undef, 13 - 4 + 1, 3);
errors_BT = Array{Float64}(undef, 13 - 4 + 1, 3);
epsilons = [];
i = 1;
c = 1;
for j in 4:13
    epsilon = (1 / 2)^j
    push!(epsilons, epsilon)

    h = c * epsilon
    nt = Int((tf - t0) / h + 1)
    println("j = ", j, ", nt = ", nt)

    # Runge Kutta (ground truth)
    x_tRK, v_tRK = runge_kutta(x_0, v_0, (t0, tf), nt, epsilon)
    v_tRK_pa = parallel_velocity.(eachcol(x_tRK), eachcol(v_tRK))
    v_tRK_pa = transpose(mapreduce(permutedims, vcat, v_tRK_pa))
    v_tRK_pe = v_tRK - v_tRK_pa

    # Standard Boris
    x_tSB, v_tSB = boris2(x_0, v_0, (t0, tf), nt, epsilon)
    v_tSB_pa = parallel_velocity.(eachcol(x_tSB), eachcol(v_tSB))
    v_tSB_pa = transpose(mapreduce(permutedims, vcat, v_tSB_pa))
    v_tSB_pe = v_tSB - v_tSB_pa

    # Explicit Filtered Boris
    x_tBEA, v_tBEA = boris_expA2(x_0, v_0, (t0, tf), nt, epsilon)
    v_tBEA_pa = parallel_velocity.(eachcol(x_tBEA), eachcol(v_tBEA))
    v_tBEA_pa = transpose(mapreduce(permutedims, vcat, v_tBEA_pa))
    v_tBEA_pe = v_tBEA - v_tBEA_pa

    # Implicit Filtered Boris
    x_tBIA, v_tBIA = boris_impA2(x_0, v_0, (t0, tf), nt, epsilon)
    v_tBIA_pa = parallel_velocity.(eachcol(x_tBIA), eachcol(v_tBIA))
    v_tBIA_pa = transpose(mapreduce(permutedims, vcat, v_tBIA_pa))
    v_tBIA_pe = v_tBIA - v_tBIA_pa

    # Two point filtered Boris
    x_tBT, v_tBT = boris_twoPA2(x_0, v_0, (t0, tf), nt, epsilon)
    v_tBT_pa = parallel_velocity.(eachcol(x_tBT), eachcol(v_tBT))
    v_tBT_pa = transpose(mapreduce(permutedims, vcat, v_tBT_pa))
    v_tBT_pe = v_tBT - v_tBT_pa

    # Global error in position
    errors_SB[i, 1] = sum(abs.(x_tRK .- x_tSB), dims=1)[nt]
    errors_BEA[i, 1] = sum(abs.(x_tRK .- x_tBEA), dims=1)[nt]
    errors_BIA[i, 1] = sum(abs.(x_tRK .- x_tBIA), dims=1)[nt]
    errors_BT[i, 1] = sum(abs.(x_tRK .- x_tBT), dims=1)[nt]

    # Global error in parallel velocities
    errors_SB[i, 2] = sum(abs.(v_tRK_pa .- v_tSB_pa), dims=1)[nt]
    errors_BEA[i, 2] = sum(abs.(v_tRK_pa .- v_tBEA_pa), dims=1)[nt]
    errors_BIA[i, 2] = sum(abs.(v_tRK_pa .- v_tBIA_pa), dims=1)[nt]
    errors_BT[i, 2] = sum(abs.(v_tRK_pa .- v_tBT_pa), dims=1)[nt]

    # Global error in perpendicular velocities
    errors_SB[i, 3] = sum(abs.(v_tRK_pe .- v_tSB_pe), dims=1)[nt]
    errors_BEA[i, 3] = sum(abs.(v_tRK_pe .- v_tBEA_pe), dims=1)[nt]
    errors_BIA[i, 3] = sum(abs.(v_tRK_pe .- v_tBIA_pe), dims=1)[nt]
    errors_BT[i, 3] = sum(abs.(v_tRK_pe .- v_tBT_pe), dims=1)[nt]

    i += 1

end

# Position errors plot
plot(epsilons, errors_BEA[:, 1], label="ExpA", xscale=:log10, yscale=:log10, marker=:square)
plot!(epsilons, errors_BIA[:, 1], label="ImpA", marker=:circle, linestyle=:dash)
plot!(epsilons, errors_SB[:, 1], label="Boris", xscale=:log10, yscale=:log10, marker=:star, linestyle=:dash)
plot!(epsilons, errors_BT[:, 1], label="Boris twoP", xscale=:log10, yscale=:log10, marker=:circle)

plot!(xlabel=L"$\log_{10}(\epsilon)$", ylabel=L"$\log_{10}($GE$)$")
plot!(title=L"The global errors of $x$ with $h=%$c c$")
savefig("errors_x.png")

# Parallel velocity errors plot
plot(epsilons, errors_BEA[:, 2], label="ExpA", xscale=:log10, yscale=:log10, marker=:square)
plot!(epsilons, errors_BIA[:, 2], label="ImpA", marker=:circle, linestyle=:dash)
plot!(epsilons, errors_SB[:, 2], label="Boris", xscale=:log10, yscale=:log10, marker=:star, linestyle=:dash)
plot!(epsilons, errors_BT[:, 2], label="Boris twoP", xscale=:log10, yscale=:log10, marker=:circle)

plot!(xlabel=L"$\log_{10}(\epsilon)$", ylabel=L"$\log_{10}($GE$)$")
plot!(title=L"The global errors of $v_{||}$ with $h=%$c c$")
savefig("errors_v_par.png")

# Perpendicular velocity errors plot
plot(epsilons, errors_BEA[:, 3], label="ExpA", xscale=:log10, yscale=:log10, marker=:square)
plot!(epsilons, errors_BIA[:, 3], label="ImpA", marker=:circle, linestyle=:dash)
plot!(epsilons, errors_SB[:, 3], label="Boris", xscale=:log10, yscale=:log10, marker=:star, linestyle=:dash)
plot!(epsilons, errors_BT[:, 3], label="Boris twoP", xscale=:log10, yscale=:log10, marker=:circle)

plot!(xlabel=L"$\log_{10}(\epsilon)$", ylabel=L"$\log_{10}($GE$)$")
plot!(title=L"The global errors of $v_{\perp}$ with $h=%$c c$")
savefig("errors_v_per.png")