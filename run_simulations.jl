"""Plots the errors in x against different epsilon = 1/2^j are displayed"""
# To run script open julia terminal and run the following commands:
# import Pkg
# Pkg.activate("BorisPusher/Boris")
# include("BorisPusher/run_simulations.jl")

using Plots

include("integrators.jl")

# initial position
x_0 = [1/3, 1/4, 1/2];
# initial velocity
v_0 = [2/5, 2/3, 1];

# numerical parameters, time
t0 = 0;
tf = 1;

# array to store the errors
error_SB = [];
error_BEA = [];
error_BIA = [];
error_BT = [];
epsilons = [];
for j in 4:13
    epsilon = (1/2)^j;
    push!(epsilons, epsilon)
    
    # h = epsilon, 4*epsilon, 16*epsilon
    h = epsilon;
    nt = Int((tf - t0) / h + 1)

    # Verbose
    println("j = ", j)
    println("epsilon = ", epsilon)
    println("h = ", h)
    println("nt = ", nt, '\n')

    # compute the trajectories
    x_tSB, v_tSB = boris(x_0, v_0, (t0, tf), nt, epsilon);
    x_tRK, v_tRK = runge_kutta(x_0, v_0, (t0, tf), nt, epsilon);
    x_tBEA, v_tBEA = boris_expA(x_0, v_0, (t0, tf), nt, epsilon);
    x_tBIA, v_tBIA = boris_impA(x_0, v_0, (t0, tf), nt, epsilon);
    x_tBT, v_tBT = boris_twoPA(x_0, v_0, (t0, tf), nt, epsilon);

    # global error
    push!(error_SB, sum(abs.(x_tRK .- x_tSB), dims=1)[nt-1])
    push!(error_BEA, sum(abs.(x_tRK .- x_tBEA), dims=1)[nt-1])
    push!(error_BIA, sum(abs.(x_tRK .- error_BIA), dims=1)[nt-1])
    push!(error_BT, sum(abs.(x_tRK .- x_tBT), dims=1)[nt-1])

end


# log-log plot
plot(epsilons, error_SB, label="Boris", xscale=:log10, yscale=:log10, marker=:circle)
plot!(epsilons, error_BEA, label="Boris expA", xscale=:log10, yscale=:log10, marker=:circle)
plot!(epsilons, error_BIA, label="Boris impA", xscale=:log10, yscale=:log10, marker=:circle)
plot!(epsilons, error_BT, label="Boris twoP", xscale=:log10, yscale=:log10, marker=:circle)

plot!(xlabel=s"$log_{10}(\epsilon)$", ylabel=s"$log_{10}($GE$)$")
plot!(title=s"Errors of $x$ with $h=\epsilon$")

savefig("errors_x_h_epsilon.png")