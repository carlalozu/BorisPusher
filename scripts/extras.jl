using LinearAlgebra
using DifferentialEquations
include("../src/utils.jl")

function B(x, epsilon)
    """Magnetic field"""
    return [-x[1], 0, x[3] + 1.0 / epsilon]
end

function E(x)
    """Mlectric field"""
    return [x[1], x[2], 0] / (x[1] .^ 2 + x[2] .^ 2)^(3 / 2)
end

function system!(du, u, p, t)
    # System of equations
    x1, x1_prime, x2, x2_prime, x3, x3_prime = u
    epsilon = p[1]

    # Equations
    du[1] = x1_prime            # x1' = x1_prime
    du[2] = x1 / (x1^2 + x2^2)^(3 / 2) + x2_prime / epsilon + x2_prime * x3 # x1''
    du[3] = x2_prime            # x2' = x2_prime
    du[4] = x2 / (x1^2 + x2^2)^(3 / 2) - x3_prime * x1 - x1_prime / epsilon - x1_prime * x3 # x2''
    du[5] = x3_prime            # x3' = x3_prime
    du[6] = x2_prime * x1       # x3''
end

function runge_kutta(x_0::Vector{Float64}, v_0::Vector{Float64}, t::Tuple{Float64,Float64}, nt::Int64, epsilon::Float64)
    """Runge-Kutta integrator"""

    # Parameters
    (t0, tf) = t

    # Initial conditions
    u0 = [x_0[1], v_0[1], x_0[2], v_0[2], x_0[3], v_0[3]]

    # Solving the system using a Runge-Kutta method
    prob = ODEProblem(system!, u0, (t0, tf), [epsilon])
    sol = solve(prob, Tsit5(),
        abstol=1e-12,
        reltol=1e-12
    )

    # Recovering the states corresponding to the times
    states = [sol(ti) for ti in range(t0, tf, nt)]

    # Extracting the positions and velocities
    x_t = [state[1:2:end] for state in states]
    x_t = mapreduce(permutedims, vcat, x_t)

    v_t = [state[2:2:end] for state in states]
    v_t = mapreduce(permutedims, vcat, v_t)

    return x_t, v_t
end
