"""Plots the errors in x against different epsilon = 1/2^j are displayed"""
# To run script open julia terminal and run the following commands:
# import Pkg
# Pkg.activate("BorisPusher/Boris")
# include("BorisPusher/run_simulations.jl")

using Plots
using DifferentialEquations
using LinearAlgebra
using LaTeXStrings
using NLsolve

include("matrix_funcs.jl")
include("extras.jl")
include("utils.jl")
include("integrators.jl")

# initial position
x_0 = [1 / 3, 1 / 4, 1 / 2];
# initial velocity
v_0 = [2 / 5, 2 / 3, 1.0];

# numerical parameters, time
t0 = 0.0;
tf = 1.0;

ji, jf = (4, 10)
# array to store the errors
errors_SB = Array{Float64}(undef, jf - ji + 1, 3)
errors_BEA = Array{Float64}(undef, jf - ji + 1, 3)
errors_BIA = Array{Float64}(undef, jf - ji + 1, 3)
errors_BT = Array{Float64}(undef, jf - ji + 1, 3)
epsilons = [];

# Define a 3x1 grid layout
plot_layout = @layout [a b c]

# Create a plot with the defined layout
p = plot(layout=plot_layout, size=(1400, 400), legend=:bottomright)

c = 1;

counter = 1;
for j in ji:jf
    global counter
    epsilon = (1 / 2)^j
    push!(epsilons, epsilon)

    h = c * epsilon
    nt = Int((tf - t0) / h + 1)
    println("j = ", j, ", nt = ", nt)

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
    # x_tBIA, _ = boris_impA(x_0, v_0, (t0, tf), nt, epsilon);
    x_tBIA, v_tBIA = boris_impA2(x_0, v_0, (t0, tf), nt, epsilon)
    v_tBIA_pa = parallel_velocity(x_tBIA[nt, :], v_tBIA[nt, :], epsilon)
    v_tBIA_pe = v_tBIA[nt, :] - v_tBIA_pa

    # Two point filtered Boris
    # x_tBT, _ = boris_twoPA(x_0, v_0, (t0, tf), nt, epsilon);
    x_tBT, v_tBT = boris_twoPA2(x_0, v_0, (t0, tf), nt, epsilon)
    v_tBT_pa = parallel_velocity(x_tBT[nt, :], v_tBT[nt, :], epsilon)
    v_tBT_pe = v_tBT[nt, :] - v_tBT_pa

    # Global error in position
    errors_SB[counter, 1] = sum(abs.(x_tRK[nt, :] .- x_tSB[nt, :]))
    errors_BEA[counter, 1] = sum(abs.(x_tRK[nt, :] .- x_tBEA[nt, :]))
    errors_BIA[counter, 1] = sum(abs.(x_tRK[nt, :] .- x_tBIA[nt, :]))
    errors_BT[counter, 1] = sum(abs.(x_tRK[nt, :] .- x_tBT[nt, :]))

    # Global error in parallel velocities
    errors_SB[counter, 2] = sum(abs.(v_tRK_pa .- v_tSB_pa))
    errors_BEA[counter, 2] = sum(abs.(v_tRK_pa .- v_tBEA_pa))
    errors_BIA[counter, 2] = sum(abs.(v_tRK_pa .- v_tBIA_pa))
    errors_BT[counter, 2] = sum(abs.(v_tRK_pa .- v_tBT_pa))

    # Global error in perpendicular velocities
    errors_SB[counter, 3] = sum(abs.(v_tRK_pe .- v_tSB_pe))
    errors_BEA[counter, 3] = sum(abs.(v_tRK_pe .- v_tBEA_pe))
    errors_BIA[counter, 3] = sum(abs.(v_tRK_pe .- v_tBIA_pe))
    errors_BT[counter, 3] = sum(abs.(v_tRK_pe .- v_tBT_pe))

    counter += 1

end

# Reference slope
slope_1 = epsilons
slope_2 = epsilons.^2

# Position errors plot
ylims = (10e-12, 10e0)
yticks = 0:-2:-10
if c > 1
    ylims = (10e-10, 10e2)
    yticks = 2:-2:-8
end
idx = 1
plot!(p[idx], xscale=:log10, yscale=:log10)
plot!(p[idx], epsilons, errors_BEA[:, 1], label="Exp-A", marker=:square)
plot!(p[idx], epsilons, errors_BIA[:, 1], label="Imp-A", marker=:circle)
plot!(p[idx], epsilons, errors_BT[:, 1], label="Two P-A", marker=:circle)
plot!(p[idx], epsilons, errors_SB[:, 1], label="Boris", marker=:star)

plot!(p[idx], xlabel=L"$\log_{10}(\epsilon)$", ylabel=L"$\log_{10}($GE$)$")
plot!(p[idx], title=L"The global errors of $x$ with $h=%$c c$")
plot!(p[idx], ylim=ylims, yticks=(10.0 .^ yticks, [L"%$x" for x in yticks]))
plot!(p[idx], epsilons, slope_1, label="slope 1", linestyle=:dash, color=:black)
plot!(p[idx], epsilons, slope_2, label="slope 2", linestyle=:dash, color=:gray)

# Parallel velocity errors plot
idx += 1
plot!(p[idx], xscale=:log10, yscale=:log10)
plot!(p[idx], epsilons, errors_BEA[:, 2], label="Exp-A", marker=:square)
plot!(p[idx], epsilons, errors_BIA[:, 2], label="Imp-A", marker=:circle)
plot!(p[idx], epsilons, errors_BT[:, 2], label="Two P-A", marker=:circle)
plot!(p[idx], epsilons, errors_SB[:, 2], label="Boris", marker=:star)

plot!(p[idx], xlabel=L"$\log_{10}(\epsilon)$", ylabel=L"$\log_{10}($GE$)$")
plot!(p[idx], title=L"The global errors of $v_{||}$ with $h=%$c c$")
plot!(p[idx], ylim=ylims, yticks=(10.0 .^ yticks, [L"%$x" for x in yticks]))
plot!(p[idx], epsilons, slope_1, label="slope 1", linestyle=:dash, color=:black)
plot!(p[idx], epsilons, slope_2, label="slope 2", linestyle=:dash, color=:gray)

# Perpendicular velocity errors plot
idx += 1
plot!(p[idx], xscale=:log10, yscale=:log10)
plot!(p[idx], epsilons, errors_BEA[:, 3], label="Exp-A", marker=:square)
plot!(p[idx], epsilons, errors_BIA[:, 3], label="Imp-A", marker=:circle)
plot!(p[idx], epsilons, errors_BT[:, 3], label="Two P-A", marker=:circle)
plot!(p[idx], epsilons, errors_SB[:, 3], label="Boris", marker=:star)

plot!(p[idx], xlabel=L"$\log_{10}(\epsilon)$", ylabel=L"$\log_{10}($GE$)$")
plot!(p[idx], title=L"The global errors of $ v_{\perp}$ with \$h=$c\$")
plot!(p[idx], ylim=ylims, yticks=(10.0 .^ yticks, [L"%$x" for x in yticks]))
plot!(p[idx], epsilons, slope_1, label="slope 1", linestyle=:dash, color=:black)
plot!(p[idx], epsilons, slope_2, label="slope 2", linestyle=:dash, color=:gray)

plot!(p, left_margin=5mm, bottom_margin=5mm, legend=:bottomright, linestyle=:dash)
plot!(p, xlim=(10e-5, 10e-2), xticks=(10.0.^(-1:-0.5:-4), [L"%$x" for x in -1:-0.5:-4]))


# Save the plot as a single image
savefig("errors_v_per.pdf")