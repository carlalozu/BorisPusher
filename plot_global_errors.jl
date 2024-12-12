# To run script open julia terminal and run the following commands:
# import Pkg
# Pkg.activate("BorisPusher/Boris")
# include("BorisPusher/plot_global_errors.jl")

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

nt_start = 75
nt_end = 1000

# Generate the incremental step pattern
steps = Int[]
current_step = 1.1
while sum(steps) < (nt_end - nt_start)
    global current_step
    append!(steps, floor(current_step))
    current_step = current_step^(1.015)
end

# Generate nt values based on the step pattern
nt_values = [nt_start]
for step in steps
    push!(nt_values, nt_values[end] + step)
end


# Arrays to store the errors
errors_SB = Array{Float64}(undef, length(nt_values), 3)
errors_BEA = Array{Float64}(undef, length(nt_values), 3)
errors_BIA = Array{Float64}(undef, length(nt_values), 3)
errors_BT = Array{Float64}(undef, length(nt_values), 3)

h_epsilons = []
i = 1
for nt in nt_values
    global i

    epsilon = (1 / 2)^10
    
    h = (tf - t0) / (nt - 1)
    push!(h_epsilons, h / epsilon)

    println("i = ", i, ", nt = ", nt, ", h/eps = ", h/epsilon)

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

# Define a 3x4 grid layout
plot_layout = @layout [
    a b c;
    d e f;
    g h i;
    j k l
]

# Create a plot with the defined layout
p = plot(layout=plot_layout, size=(1400, 1600))
plot!(p, yscale=:log10, legend=false)
plot!(p, xlabel=L"$h/\epsilon$", ylabel=L"$\log_{10}(GE)$")
plot!(ylims = (10e-9, 10e0), yticks=(10.0.^(0:-2:-6), [L"%$x" for x in 0:-2:-8]))
plot!(xlims = (2, 14), xticks=(pi*(1:1:4), [L"%$x \pi" for x in 1:1:4]))
plot!(guidefontsize=16, tickfontsize=14)

# Standard Boris
idx = 1
plot!(p[idx], h_epsilons, errors_SB[:, 1], linewidth=3, color=:dodgerblue)
plot!(p[idx], title=L"The global errors of $x$ for Boris")
idx += 1
plot!(p[idx], h_epsilons, errors_SB[:, 2], linewidth=3, color=:dodgerblue)
plot!(p[idx], title=L"The global errors of $v_{||}$ for Boris")
idx += 1
plot!(p[idx], h_epsilons, errors_SB[:, 3], linewidth=3, color=:dodgerblue)
plot!(p[idx], title=L"The global errors of $v_{\perp}$ for Boris")

# Explicit Filtered Boris
idx += 1
plot!(p[idx], h_epsilons, errors_BEA[:, 1], linewidth=3, color=:dodgerblue)
plot!(p[idx], title=L"The global errors of $x$ for Exp-A")
idx += 1
plot!(p[idx], h_epsilons, errors_BEA[:, 2], linewidth=3, color=:dodgerblue)
plot!(p[idx], title=L"The global errors of $v_{||}$ for Exp-A")
idx += 1
plot!(p[idx], h_epsilons, errors_BEA[:, 3], linewidth=3, color=:dodgerblue)
plot!(p[idx], title=L"The global errors of $v_{\perp}$ for Exp-A")

# Implicit Filtered Boris
idx += 1
plot!(p[idx], h_epsilons, errors_BIA[:, 1], linewidth=3, color=:dodgerblue)
plot!(p[idx], title=L"The global errors of $x$ for Imp-A")
idx += 1
plot!(p[idx], h_epsilons, errors_BIA[:, 2], linewidth=3, color=:dodgerblue)
plot!(p[idx], title=L"The global errors of $v_{||}$ for Imp-A")
idx += 1
plot!(p[idx], h_epsilons, errors_BIA[:, 3], linewidth=3, color=:dodgerblue)
plot!(p[idx], title=L"The global errors of $v_{\perp}$ for Imp-A")

# Two point filtered Boris
idx += 1
plot!(p[idx], h_epsilons, errors_BT[:, 1], linewidth=3, color=:dodgerblue)
plot!(p[idx], title=L"The global errors of $x$ for Two P-A")
idx += 1
plot!(p[idx], h_epsilons, errors_BT[:, 2], linewidth=3, color=:dodgerblue)
plot!(p[idx], title=L"The global errors of $v_{||}$ for Two P-A")
idx += 1
plot!(p[idx], h_epsilons, errors_BT[:, 3], linewidth=3, color=:dodgerblue)
plot!(p[idx], title=L"The global errors of $v_{\perp}$ for Two P-A")


plot!(p, right_margin=5mm, left_margin=5mm, top_margin=5mm, bottom_margin=5mm)

# Save the combined plot as a single image
savefig("errors_2.pdf")
