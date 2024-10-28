using DifferentialEquations
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

function boris(x_0, v_0, t::Tuple, nt::Int, epsilon)
    # standard Boris integrator

    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)

    # Arrays to store the state
    x_t = Array{Float64}(undef, 3, nt)
    v_t = Array{Float64}(undef, 3, nt)

    v = v_0
    x = x_0

    # Initial half-step for velocity
    v = v - (cross(v, B(x, epsilon)) + E(x))*h/2
    for i in 1:nt
        # Store the position and velocity
        x_t[:, i] = x
        v_t[:, i] = v

        # Half step of velocity due to electric field
        v_minus = v .+ h / 2 * E(x)

        # Magnetic field rotation
        t_ = h / 2 * B(x, epsilon)
        v_prime = v_minus .+ cross(v_minus, t_)
        s = 2 / (1 + dot(t_, t_))
        v_plus = v_minus .+ s * cross(v_prime, t_)

        # Half step of the velocity again
        v = v_plus .+ h / 2 * E(x)

        # Full step of the position
        x = x .+ h * v

    end

    return x_t, v_t
end

function runge_kutta(x_0, v_0, t::Tuple, nt::Int, epsilon)
    # Runge-Kutta integrator

    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)
    time = t0:h:tf
    u0 = [x_0[1], v_0[1], x_0[2], v_0[2], x_0[3], v_0[3]]

    # Solving the system using a Runge-Kutta method
    prob = ODEProblem(system!, u0, (t0, tf), [epsilon])
    sol = solve(prob, Vern7())

    # Recovering the states corresponding to the times
    u_t = sol(time).u

    # Extracting the positions and velocities
    x_t = [[u_t[i][1], u_t[i][3], u_t[i][5]] for i in 1:nt]
    x_t = hcat(x_t...)

    v_t = [[u_t[i][2], u_t[i][4], u_t[i][6]] for i in 1:nt]
    v_t = hcat(v_t...)

    return x_t, v_t
end

function boris_expA(x_0, v_0, t::Tuple, nt::Int, epsilon)
    # standard Boris integrator

    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)

    # Arrays to store the state
    x_t = Array{Float64}(undef, 3, nt)
    v_t = Array{Float64}(undef, 3, nt)

    v = v_0
    x = x_0

    # Initial half-step for velocity
    v = v - (cross(v, B(x, epsilon)) + E(x))*h/2
    for i in 1:nt
        # Store the position and velocity
        x_t[:, i] = x
        v_t[:, i] = v

        b_hat = B_hat(B(x, epsilon))
        v_plus = v + h / 2 * Psi(h,  b_hat) * E(x)

        v_minus = exp(-h*b_hat) * v_plus

        v = v_minus + h / 2 * Psi(h,  b_hat) * E(x)

        # Full step of the position
        x = x + h * v

    end
    return x_t, v_t
end

function boris_impA(x_0, v_0, t::Tuple, nt::Int, epsilon)
    # standard Boris integrator

    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)

    # Arrays to store the state
    x_t = Array{Float64}(undef, 3, nt)
    v_t = Array{Float64}(undef, 3, nt)

    v = v_0
    x = x_0

    # Initial half-step for velocity
    v = v - (cross(v, B(x, epsilon)) + E(x))*h/2
    for i in 1:nt
        # Store the position and velocity
        x_t[:, i] = x # x^{n}
        v_t[:, i] = v # v^{n-1/2}

        B_n = B(x, epsilon)
        B_hat_n = B_hat(B_n)

        # v^{n-1/2}_{+}
        v_plus = v + h / 2 * Psi(h,  B_hat_n) * E(x)

        theta_n = theta(h, B_n)
        
        # ﬁxed-point iteration to solve for x_bar_n
        x_bar_n = x
        tol = 1
        v_minus = 0
        while tol > 1e-8
            b_bar_n = B(x_bar_n, epsilon)
            # v^{n-1/2}_{-} = exp(-h*B_hat_bar_n)) * v^{n-1/2}_{+}
            v_minus = exp(-h*B_hat(b_bar_n)) * v_plus

            # n^n
            vn = Phi_1(h, B_hat(b_bar_n)) * 1/2 * (v_minus + v_plus) - h * Gamma(h, B_hat_n) * E(x)

            x_c = x_center(x, vn, B_n)
            x_bar_n_ = x_bar(theta_n, x, x_c)

            tol = norm(x_bar_n - x_bar_n_) / norm(x_bar_n_)
            x_bar_n = x_bar_n_
        end

        # v^{n+1/2}
        v = v_minus + h / 2 * Psi(h,  B_hat_n) * E(x)

        # Full step of the position x^{n+1}
        x = x + h * v

    end
    return x_t, v_t
end