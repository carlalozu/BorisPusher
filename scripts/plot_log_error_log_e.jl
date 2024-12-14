# To run script open julia terminal and run the following commands:
# import Pkg
# Pkg.activate("Boris")
# include("scripts/plot_log_error_log_e.jl")

using Plots
using DifferentialEquations
using LinearAlgebra
using NLsolve
using LaTeXStrings
using Measures

include("extras.jl")
include("../src/matrix_funcs.jl")
include("../src/utils.jl")
include("../src/integrators_one_step_map.jl")

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
p = plot(layout=plot_layout, size=(1600, 1400), link=:x, legend=:bottomright, margin = 5mm)
plot!(p, xscale=:log10, yscale=:log10)
plot!(p, guidefontsize=16, tickfontsize=14)
plot!(p, xlabel=L"$\log_{10}(\epsilon)$", ylabel=L"$\log_{10}(GE)$")
plot!(p, xlim=(10e-5, 10e-2), xticks=(10.0.^(-1:-0.5:-4), [L"%$x" for x in -1:-0.5:-4]))

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
    println("c = ", c)
    for j in ji:jf
        epsilon = (1 / 2)^j
        push!(epsilons, epsilon)

        h = c * epsilon
        nt = Int((tf - t0) / h + 1)
        println("j = ", j, ", nt = ", nt, ", h = ", h)

        # Runge Kutta (ground truth)
        x_tRK, v_tRK = runge_kutta(x_0, v_0, (t0, tf), nt, epsilon)
        v_tRK_pa = parallel_velocity(x_tRK[nt, :], v_tRK[nt, :], epsilon)
        v_tRK_pe = v_tRK[nt, :] - v_tRK_pa

        # Standard Boris
        x_tSB, v_tSB = boris2(x_0, v_0, (t0, tf), nt, epsilon)
        v_tSB_pa = parallel_velocity(x_tSB[nt, :], v_tSB[nt, :], epsilon)
        v_tSB_pe = v_tSB[nt, :] - v_tSB_pa

        # Explicit Filtered Boris
        x_tBEA, v_tBEA = boris_expA2(x_0, v_0, (t0, tf), nt, epsilon)
        v_tBEA_pa = parallel_velocity(x_tBEA[nt, :], v_tBEA[nt, :], epsilon)
        v_tBEA_pe = v_tBEA[nt, :] - v_tBEA_pa

        # Implicit Filtered Boris
        x_tBIA, v_tBIA = boris_impA2(x_0, v_0, (t0, tf), nt, epsilon)
        v_tBIA_pa = parallel_velocity(x_tBIA[nt, :], v_tBIA[nt, :], epsilon)
        v_tBIA_pe = v_tBIA[nt, :] - v_tBIA_pa

        # Two point filtered Boris
        x_tBT, v_tBT = boris_twoPA2(x_0, v_0, (t0, tf), nt, epsilon)
        v_tBT_pa = parallel_velocity(x_tBT[nt, :], v_tBT[nt, :], epsilon)
        v_tBT_pe = v_tBT[nt, :] - v_tBT_pa

        # Global error in position
        errors_SB[i, 1] = sum(abs.(x_tRK[nt, :] .- x_tSB[nt, :]))
        errors_BEA[i, 1] = sum(abs.(x_tRK[nt, :] .- x_tBEA[nt, :]))
        errors_BIA[i, 1] = sum(abs.(x_tRK[nt, :] .- x_tBIA[nt, :]))
        errors_BT[i, 1] = sum(abs.(x_tRK[nt, :] .- x_tBT[nt, :]))

        # Global error in parallel velocities
        errors_SB[i, 2] = sum(abs.(v_tRK_pa .- v_tSB_pa))
        errors_BEA[i, 2] = sum(abs.(v_tRK_pa .- v_tBEA_pa))
        errors_BIA[i, 2] = sum(abs.(v_tRK_pa .- v_tBIA_pa))
        errors_BT[i, 2] = sum(abs.(v_tRK_pa .- v_tBT_pa))

        # Global error in perpendicular velocities
        errors_SB[i, 3] = sum(abs.(v_tRK_pe .- v_tSB_pe))
        errors_BEA[i, 3] = sum(abs.(v_tRK_pe .- v_tBEA_pe))
        errors_BIA[i, 3] = sum(abs.(v_tRK_pe .- v_tBIA_pe))
        errors_BT[i, 3] = sum(abs.(v_tRK_pe .- v_tBT_pe))

        i += 1
    end

    # Plot limits and ticks
    ylims = (10e-12, 10e0)
    yticks = 0:-2:-10
    if c > 1
        ylims = (10e-10, 10e2)
        yticks = 2:-2:-8
    end

    # Reference slope
    slope_1 = epsilons
    slope_2 = epsilons.^2

    idx += 1
    plot!(p[idx], epsilons, errors_BEA[:, 1], label="Exp-A", marker=:square, markerstrokewidth=0, linewidth=3)
    plot!(p[idx], epsilons, errors_BIA[:, 1], label="Imp-A", marker=:circle, markerstrokewidth=0, linewidth=3)
    plot!(p[idx], epsilons, errors_BT[:, 1], label="Two P-A", marker=:circle, markerstrokewidth=0, linewidth=3)
    plot!(p[idx], epsilons, errors_SB[:, 1], label="Boris", marker=:star, markerstrokewidth=0, linewidth=3)

    plot!(p[idx], title=L"$x$")
    plot!(p[idx], ylim=ylims, yticks=(10.0.^yticks, [L"%$x" for x in yticks]))
    plot!(p[idx], epsilons, slope_2, label="slope 2", linestyle=:dash, color=:gray)
    plot!(p[idx], epsilons, slope_1, label="slope 1", linestyle=:dash, color=:black)

    # Parallel velocity errors plot
    idx += 1
    plot!(p[idx], epsilons, errors_BEA[:, 2], label="Exp-A", marker=:square, markerstrokewidth=0, linewidth=3)
    plot!(p[idx], epsilons, errors_BIA[:, 2], label="Imp-A", marker=:circle, markerstrokewidth=0, linewidth=3)
    plot!(p[idx], epsilons, errors_BT[:, 2], label="Two P-A", marker=:circle, markerstrokewidth=0, linewidth=3)
    plot!(p[idx], epsilons, errors_SB[:, 2], label="Boris", marker=:star, markerstrokewidth=0, linewidth=3)

    plot!(p[idx], title=L"$v_{||}$")
    plot!(p[idx], ylim=ylims, yticks=(10.0.^yticks, [L"%$x" for x in yticks]))
    plot!(p[idx], epsilons, slope_2, label="slope 2", linestyle=:dash, color=:gray)
    plot!(p[idx], epsilons, slope_1, label="slope 1", linestyle=:dash, color=:black)

    # Perpendicular velocity errors plot
    idx += 1
    plot!(p[idx], epsilons, errors_BEA[:, 3], label="Exp-A", marker=:square, markerstrokewidth=0, linewidth=3)
    plot!(p[idx], epsilons, errors_BIA[:, 3], label="Imp-A", marker=:circle, markerstrokewidth=0, linewidth=3)
    plot!(p[idx], epsilons, errors_BT[:, 3], label="Two P-A", marker=:circle, markerstrokewidth=0, linewidth=3)
    plot!(p[idx], epsilons, errors_SB[:, 3], label="Boris", marker=:star, markerstrokewidth=0, linewidth=3)

    plot!(p[idx], title=L"$v_{\perp}$")
    plot!(p[idx], ylim=ylims, yticks=(10.0.^yticks, [L"%$x" for x in yticks]))
    plot!(p[idx], epsilons, slope_2, label="slope 2", linestyle=:dash, color=:gray)
    plot!(p[idx], epsilons, slope_1, label="slope 1", linestyle=:dash, color=:black)

end

# Save the combined plot as a single image
savefig("figures/log_error_log_e.pdf")
