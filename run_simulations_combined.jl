# To run script open julia terminal and run the following commands:
# import Pkg
# Pkg.activate("BorisPusher/Boris")
# include("BorisPusher/run_simulations_combined.jl")

using Plots
using DifferentialEquations
using LinearAlgebra
using NLsolve
using LaTeXStrings
using Measures

include("matrix_funcs.jl")
include("extras.jl")
include("utils.jl")
include("integrators.jl")
include("int_one_step_map.jl")

# Initial position
x_0 = [1 / 3, 1 / 4, 1 / 2];
# Initial velocity
v_0 = [2 / 5, 2 / 3, 1.0];

# Numerical parameters, time
t0 = 0.0;
tf = 1.0;

# Define a 3x3 grid layout
plot_layout = @layout [a b c;
    d e f;
    g h i]

# Create a plot with the defined layout
p = plot(layout=plot_layout, size=(1400, 1000), link=:x, legend=:bottomright)

ji, jf = (4, 13)
for (idx, c) in enumerate([1, 4, 16])
    # Arrays to store the errors
    errors_SB = Array{Float64}(undef, jf - ji + 1, 3)
    errors_BEA = Array{Float64}(undef, jf - ji + 1, 3)
    errors_BIA = Array{Float64}(undef, jf - ji + 1, 3)
    errors_BT = Array{Float64}(undef, jf - ji + 1, 3)
    epsilons = []

    idx -= 1
    idx *= 3

    i = 1
    for j in ji:jf
        epsilon = (1 / 2)^j
        push!(epsilons, epsilon)

        h = c * epsilon
        nt = Int((tf - t0) / h + 1)
        println("j = ", j, ", nt = ", nt)

        # Runge Kutta (ground truth)
        x_tRK, v_tRK = runge_kutta(x_0, v_0, (t0, tf), nt, epsilon)
        v_tRK_pa = parallel_velocity.(eachrow(x_tRK), eachrow(v_tRK), epsilon)
        v_tRK_pa = mapreduce(permutedims, vcat, v_tRK_pa)
        v_tRK_pe = v_tRK - v_tRK_pa

        # Standard Boris
        x_tSB, v_tSB = boris2(x_0, v_0, (t0, tf), nt, epsilon)
        v_tSB_pa = parallel_velocity.(eachrow(x_tSB), eachrow(v_tSB), epsilon)
        v_tSB_pa = mapreduce(permutedims, vcat, v_tSB_pa)
        v_tSB_pe = v_tSB - v_tSB_pa

        # Explicit Filtered Boris
        x_tBEA, v_tBEA = boris_expA2(x_0, v_0, (t0, tf), nt, epsilon)
        v_tBEA_pa = parallel_velocity.(eachrow(x_tBEA), eachrow(v_tBEA), epsilon)
        v_tBEA_pa = mapreduce(permutedims, vcat, v_tBEA_pa)
        v_tBEA_pe = v_tBEA - v_tBEA_pa

        # Implicit Filtered Boris
        x_tBIA, v_tBIA = boris_impA2(x_0, v_0, (t0, tf), nt, epsilon)
        v_tBIA_pa = parallel_velocity.(eachrow(x_tBIA), eachrow(v_tBIA), epsilon)
        v_tBIA_pa = mapreduce(permutedims, vcat, v_tBIA_pa)
        v_tBIA_pe = v_tBIA - v_tBIA_pa

        # Two point filtered Boris
        x_tBT, v_tBT = boris_twoPA2(x_0, v_0, (t0, tf), nt, epsilon)
        v_tBT_pa = parallel_velocity.(eachrow(x_tBT), eachrow(v_tBT), epsilon)
        v_tBT_pa = mapreduce(permutedims, vcat, v_tBT_pa)
        v_tBT_pe = v_tBT - v_tBT_pa

        # Global error in position
        errors_SB[i, 1] = sum(abs.(x_tRK .- x_tSB), dims=2)[nt]
        errors_BEA[i, 1] = sum(abs.(x_tRK .- x_tBEA), dims=2)[nt]
        errors_BIA[i, 1] = sum(abs.(x_tRK .- x_tBIA), dims=2)[nt]
        errors_BT[i, 1] = sum(abs.(x_tRK .- x_tBT), dims=2)[nt]

        # Global error in parallel velocities
        errors_SB[i, 2] = sum(abs.(v_tRK_pa .- v_tSB_pa), dims=2)[nt]
        errors_BEA[i, 2] = sum(abs.(v_tRK_pa .- v_tBEA_pa), dims=2)[nt]
        errors_BIA[i, 2] = sum(abs.(v_tRK_pa .- v_tBIA_pa), dims=2)[nt]
        errors_BT[i, 2] = sum(abs.(v_tRK_pa .- v_tBT_pa), dims=2)[nt]

        # Global error in perpendicular velocities
        errors_SB[i, 3] = sum(abs.(v_tRK_pe .- v_tSB_pe), dims=2)[nt]
        errors_BEA[i, 3] = sum(abs.(v_tRK_pe .- v_tBEA_pe), dims=2)[nt]
        errors_BIA[i, 3] = sum(abs.(v_tRK_pe .- v_tBIA_pe), dims=2)[nt]
        errors_BT[i, 3] = sum(abs.(v_tRK_pe .- v_tBT_pe), dims=2)[nt]

        i += 1
    end

    # Position errors plot
    idx += 1
    plot!(p[idx], xscale=:log10, yscale=:log10)
    plot!(p[idx], epsilons, errors_BEA[:, 1], label="ExpA", marker=:square)
    plot!(p[idx], epsilons, errors_BIA[:, 1], label="ImpA", marker=:circle, linestyle=:dash)
    plot!(p[idx], epsilons, errors_SB[:, 1], label="Boris", marker=:star, linestyle=:dash)
    plot!(p[idx], epsilons, errors_BT[:, 1], label="Boris twoP", marker=:circle)

    plot!(p[idx], xlabel=L"$\log_{10}(\epsilon)$", ylabel=L"$\log_{10}($GE$)$")
    plot!(p[idx], title=L"The global errors of $x$ with $h=%$c c$")

    # Parallel velocity errors plot
    idx += 1
    plot!(p[idx], xscale=:log10, yscale=:log10)
    plot!(p[idx], epsilons, errors_BEA[:, 2], label="ExpA", marker=:square)
    plot!(p[idx], epsilons, errors_BIA[:, 2], label="ImpA", marker=:circle, linestyle=:dash)
    plot!(p[idx], epsilons, errors_SB[:, 2], label="Boris", marker=:star, linestyle=:dash)
    plot!(p[idx], epsilons, errors_BT[:, 2], label="Boris twoP", marker=:circle)

    plot!(p[idx], xlabel=L"$\log_{10}(\epsilon)$", ylabel=L"$\log_{10}($GE$)$")
    plot!(p[idx], title=L"The global errors of $v_{||}$ with $h=%$c c$")

    # Perpendicular velocity errors plot
    idx += 1
    plot!(p[idx], xscale=:log10, yscale=:log10)
    plot!(p[idx], epsilons, errors_BEA[:, 3], label="ExpA", marker=:square)
    plot!(p[idx], epsilons, errors_BIA[:, 3], label="ImpA", marker=:circle, linestyle=:dash)
    plot!(p[idx], epsilons, errors_SB[:, 3], label="Boris", marker=:star, linestyle=:dash)
    plot!(p[idx], epsilons, errors_BT[:, 3], label="Boris twoP", marker=:circle)

    plot!(p[idx], xlabel=L"$\log_{10}(\epsilon)$", ylabel=L"$\log_{10}($GE$)$")
    plot!(p[idx], title=L"The global errors of $v_{\perp}$ with $h=%$c c$")

end

plot!(p, left_margin=5mm, bottom_margin=5mm)
# Save the combined plot as a single image
savefig("errors_x_h_epsilon_combined.png")
