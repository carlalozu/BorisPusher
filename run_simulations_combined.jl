using Plots
using DifferentialEquations
using LinearAlgebra
using NLsolve

include("matrix_funcs.jl")
include("extras.jl")
include("utils.jl")
include("integrators.jl")

# Initial position
x_0 = [1/3, 1/4, 1/2];
# Initial velocity
v_0 = [2/5, 2/3, 1];

# Numerical parameters, time
t0 = 0;
tf = 1;

# Create a figure layout for subplots
plot_layout = @layout [a b c]  # Three horizontal subplots
p = plot(layout=plot_layout, size=(1200, 400))

for (idx, c) in enumerate([1, 4, 16])
    # Arrays to store the errors
    error_SB = [];
    error_BEA = [];
    error_BIA = [];
    error_BT = [];
    epsilons = [];
    for j in 4:13
        epsilon = (1/2)^j;
        push!(epsilons, epsilon)

        # h = epsilon, 4*epsilon, 16*epsilon
        h = c*epsilon;
        nt = Int((tf - t0) / h + 1)
        println("j = ", j, ", nt = ", nt)

        x_tRK, v_tRK = runge_kutta(x_0, v_0, (t0, tf), nt, epsilon);
        x_tSB, v_tSB = boris(x_0, v_0, (t0, tf), nt, epsilon);
        x_tBEA, v_tBEA = boris_expA(x_0, v_0, (t0, tf), nt, epsilon);
        x_tBIA, v_tBIA = boris_impA_2(x_0, v_0, (t0, tf), nt, epsilon);
        x_tBT, v_tBT = boris_twoPA_2(x_0, v_0, (t0, tf), nt, epsilon);

        # Mean square error
        push!(error_SB, sum(abs.(x_tRK .- x_tSB), dims=1)[nt-1])
        push!(error_BEA, sum(abs.(x_tRK .- x_tBEA), dims=1)[nt-1])
        push!(error_BIA, sum(abs.(x_tRK .- x_tBIA), dims=1)[nt-1])
        push!(error_BT, sum(abs.(x_tRK .- x_tBT), dims=1)[nt-1])
    end

    # Log-log plot for each c
    plot!(p[idx], epsilons, error_SB, label="Boris", xscale=:log10, yscale=:log10, marker=:circle)
    plot!(p[idx], epsilons, error_BEA, label="Boris expA", xscale=:log10, yscale=:log10, marker=:circle)
    plot!(p[idx], epsilons, error_BIA, label="Boris impA", xscale=:log10, yscale=:log10, marker=:circle)
    plot!(p[idx], epsilons, error_BT, label="Boris twoP", xscale=:log10, yscale=:log10, marker=:circle)

    # Set subplot title and axis labels
    plot!(p[idx], xlabel=s"$log_{10}(\epsilon)$", ylabel=s"$log_{10}($GE$)$")
    plot!(p[idx], title=s"Errors of $x$ with $h=$ $(c=*$c$)$")
end

# Save the combined plot as a single image
savefig("errors_x_h_epsilon_combined.png")
