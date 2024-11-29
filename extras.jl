using LinearAlgebra
include("utils.jl")


function leapfrog(x_0, v_0, t::Tuple, nt::Int, epsilon)
    # Leapfrog integrator

    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)

    # Arrays to store the state
    x_t = Array{Float64}(undef, 3, nt)
    v_t = Array{Float64}(undef, 3, nt)

    x = x_0
    v = v_0

    # Initial half-step for velocity
    v = v + 0.5 * h * (cross(v, B(x, epsilon)) + E(x))
    for i in 1:nt
        # Store the current position and velocity
        x_t[:, i] = x
        v_t[:, i] = v

        # Update the position by a full step
        x = x + v * h

        # Update the velocity by a full step
        v = v + h * (cross(v, B(x, epsilon)) + E(x))
    end

    return x_t, v_t
end

function euler(x_0, v_0, t::Tuple, nt::Int, epsilon)
    # euler integrator from analytical mechanics

    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)

    # Arrays to store the state
    x_t = Array{Float64}(undef, 3, nt)
    v_t = Array{Float64}(undef, 3, nt)

    v = v_0
    x = x_0
    for i in 1:nt
        # Store the position and velocity
        x_t[:, i] = x
        v_t[:, i] = v

        a = cross(v, B(x, epsilon)) + E(x)
        v = v + a * h
        x = x + v * h
    end

    return x_t, v_t
end
