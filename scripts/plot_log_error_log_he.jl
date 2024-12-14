# To run script open julia terminal and run the following commands:
# import Pkg
# Pkg.activate("Boris")
# include("scripts/plot_global_errors.jl")

using Plots
using LaTeXStrings
using Measures

include("extras.jl")
include("../src/rodriguez.jl") # You can decide to use the rodriguez.jl file or matrix_funcs.jl
include("../src/utils.jl")
include("../src/integrators_one_step_map.jl")

# Initial position
x_0 = [1 / 3, 1 / 4, 1 / 2];
# Initial velocity
v_0 = [2 / 5, 2 / 3, 1.0];

# Numerical parameters, time
t0 = 0.0;
tf = 1.0;

# k or number of timesteps
k_start = 60
k_end = 600
k_values = range(k_start, k_end)

# Arrays to store the errors
errors_SB = Array{Float64}(undef, length(k_values), 3)
errors_BEA = Array{Float64}(undef, length(k_values), 3)
errors_BIA = Array{Float64}(undef, length(k_values), 3)
errors_BT = Array{Float64}(undef, length(k_values), 3)

h_epsilons = []
i = 1
for k in k_values
    global i

    epsilon = (1 / 2)^10
    
    nt = k + 1
    h = (tf - t0) / k
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
p = plot(layout=plot_layout, size=(1400, 1600), margin = 5Plots.mm)
plot!(p, yscale=:log10, legend=:bottomright)
plot!(p, xlabel=L"$h/\epsilon$", ylabel=L"$\log_{10}(GE)$")
plot!(ylims = (10e-9, 10e0), yticks=(10.0.^(0:-2:-6), [L"%$x" for x in 0:-2:-8]))
plot!(xlims = (2, 14), xticks=(pi*(1:1:4), [L"%$x \pi" for x in 1:1:4]))
plot!(guidefontsize=16, tickfontsize=14)

# Standard Boris
idx = 1
plot!(p[idx], h_epsilons, errors_SB[:, 1], linewidth=3, color=:dodgerblue, label="Boris")
plot!(p[idx], title=L"$x$")
idx += 1
plot!(p[idx], h_epsilons, errors_SB[:, 2], linewidth=3, color=:dodgerblue, label="Boris")
plot!(p[idx], title=L"$v_{||}$")
idx += 1
plot!(p[idx], h_epsilons, errors_SB[:, 3], linewidth=3, color=:dodgerblue, label="Boris")
plot!(p[idx], title=L"$v_{\perp}$")

# Explicit Filtered Boris
idx += 1
plot!(p[idx], h_epsilons, errors_BEA[:, 1], linewidth=3, color=:dodgerblue, label="Exp-A")
plot!(p[idx], title=L"$x$")
idx += 1
plot!(p[idx], h_epsilons, errors_BEA[:, 2], linewidth=3, color=:dodgerblue, label="Exp-A")
plot!(p[idx], title=L"$v_{||}$")
idx += 1
plot!(p[idx], h_epsilons, errors_BEA[:, 3], linewidth=3, color=:dodgerblue, label="Exp-A")
plot!(p[idx], title=L"$v_{\perp}$")

# Implicit Filtered Boris
idx += 1
plot!(p[idx], h_epsilons, errors_BIA[:, 1], linewidth=3, color=:dodgerblue, label="Imp-A")
plot!(p[idx], title=L"$x$")
idx += 1
plot!(p[idx], h_epsilons, errors_BIA[:, 2], linewidth=3, color=:dodgerblue, label="Imp-A")
plot!(p[idx], title=L"$v_{||}$")
idx += 1
plot!(p[idx], h_epsilons, errors_BIA[:, 3], linewidth=3, color=:dodgerblue, label="Imp-A")
plot!(p[idx], title=L"$v_{\perp}$")

# Two point filtered Boris
idx += 1
plot!(p[idx], h_epsilons, errors_BT[:, 1], linewidth=3, color=:dodgerblue, label="Two P-A")
plot!(p[idx], title=L"$x$")
idx += 1
plot!(p[idx], h_epsilons, errors_BT[:, 2], linewidth=3, color=:dodgerblue, label="Two P-A")
plot!(p[idx], title=L"$v_{||}$")
idx += 1
plot!(p[idx], h_epsilons, errors_BT[:, 3], linewidth=3, color=:dodgerblue, label="Two P-A")
plot!(p[idx], title=L"$v_{\perp}$")


plot!(p, right_margin=5mm, left_margin=5mm, top_margin=5mm, bottom_margin=5mm)

# Save the combined plot as a single image
savefig("figures/log_error_log_he.pdf")
