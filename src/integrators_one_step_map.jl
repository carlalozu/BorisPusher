using LinearAlgebra
using NLsolve

function boris2(x_0::Vector, v_0::Vector, t::Tuple, nt::Int, B::Function, E::Function)
    """Standard Boris integrator"""
    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)

    # Arrays to store the states
    x_t = Array{Float64}(undef, nt, 3)
    v_t = Array{Float64}(undef, nt, 3)

    v = v_0
    x = x_0

    # Initial half-step for velocity
    v_ = v - (cross(v, B(x)) + E(x)) * h / 2
    for n in 1:nt
        # Store the position and velocity at full time step
        x_t[n, :] = x # x^n
        v_t[n, :] = v # v^n

        E_n = E(x)
        B_n = B(x)

        # Half step of velocity due to electric field
        v_minus = v_ .+ h / 2 * E_n

        # Magnetic field rotation
        t_ = h / 2 * B_n
        s = 2 * t_ / (1 + dot(t_, t_))

        v_prime = v_minus .+ cross(v_minus, t_)
        v_plus = v_minus .+ cross(v_prime, s)

        # Half step of the velocity again
        v_ = v_plus .+ h / 2 * E_n

        # Full step of the position
        x = x .+ h * v_

        # Recover velocity from half step velocity
        v = inv(I + h / 2 * hat(B(x))) * (v_ + E(x) * h / 2)

    end
    return x_t, v_t
end

function boris_expA2(x_0::Vector, v_0::Vector, t::Tuple, nt::Int64, B::Function, E::Function)
    """Explicit filtered Boris integrator one-step map"""
    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)

    # Arrays to store the states
    x_t = Array{Float64}(undef, nt, 3)
    v_t = Array{Float64}(undef, nt, 3)

    x = x_0
    v = v_0

    for n in 1:nt
        # Store the position and velocity at full time step
        x_t[n, :] = x # x^n
        v_t[n, :] = v # v^n

        E_n = E(x)
        B_n = B(x)

        # Full step of the position x^{n+1}
        x_ = x + h * Phi_pm_(B_n, h, +1) * v + h^2 / 2 * Psi_pm_(B_n, B_n, h, +1) * E_n

        function equation_imp(v_)
            # At x+1 position
            B_n_ = B(x_)

            lhs = Phi_pm_(B_n_, h, -1) * v_

            term_1 = Phi_pm_(B_n, h, +1) * v
            term_2 = h / 2 * Psi_pm_(B_n_, B_n_, h, -1) * E(x_)
            term_3 = h / 2 * Psi_pm_(B_n, B_n, h, +1) * E(x)

            rhs = term_1 + term_2 + term_3
            return lhs - rhs
        end

        v = nlsolve(equation_imp, v).zero
        x = x_

    end
    return x_t, v_t
end

function boris_impA2(x_0::Vector, v_0::Vector, t::Tuple, nt::Int, B::Function, E::Function)
    """Implicit filtered Boris integrator"""
    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)

    # Arrays to store the states
    x_t = Array{Float64}(undef, nt, 3)
    v_t = Array{Float64}(undef, nt, 3)

    x = x_0
    v = v_0

    for n in 1:nt
        # Store the position and velocity at full time step
        x_t[n, :] = x # x^n
        v_t[n, :] = v # v^n

        E_n = E(x)
        B_n = B(x)

        x_c = x_center(x, v, B_n)
        theta_n = theta(h * norm(B_n))
        x_bar_n = x_bar(theta_n, x, x_c)
        B_bar_n = B(x_bar_n)

        # Full step of the position x^{n+1}
        x_ = x + h * Phi_pm_(B_bar_n, h, +1) * v + h^2 / 2 * Psi_pm_(B_n, B_bar_n, h, +1) * E_n

        function equation_imp(v_::Vector)
            # At n+1 timestep
            B_n_ = B(x_)
            x_c_ = x_center(x_, v_, B_n_)
            theta_n_ = theta(h * norm(B_n_))
            x_bar_n_ = x_bar(theta_n_, x_, x_c_)
            B_bar_n_ = B(x_bar_n_)

            lhs = Phi_pm_(B_bar_n_, h, -1) * v_

            # At n timestep
            term_1 = Phi_pm_(B_bar_n, h, +1) * v
            term_2 = h / 2 * Psi_pm_(B_n_, B_bar_n_, h, -1) * E(x_)
            term_3 = h / 2 * Psi_pm_(B_n, B_bar_n, h, +1) * E(x)

            rhs = term_1 + term_2 + term_3
            return lhs - rhs
        end

        v = nlsolve(equation_imp, v, ftol=1e-16, xtol=1e-16).zero
        x = x_

    end
    return x_t, v_t
end

function boris_twoPA2(x_0::Vector, v_0::Vector, t::Tuple, nt::Int, B::Function, E::Function)
    """Implicit filtered Boris integrator"""
    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)

    # Arrays to store the states
    x_t = Array{Float64}(undef, nt, 3)
    v_t = Array{Float64}(undef, nt, 3)

    x = x_0
    v = v_0

    for n in 1:nt
        # Store the position and velocity at full time step
        x_t[n, :] = x # x^n
        v_t[n, :] = v # v^n

        E_n = E(x)
        B_n = B(x)

        x_c = x_center(x, v, B_n)
        B_c = B(x_c)

        # Full step of the position x^{n+1}
        x = x + h * Phi_pm(B_n, B_c, h, +1) * v + h^2 / 2 * Psi_pm(B_n, B_c, h, +1) * E_n

        function equation_twop(v_::Vector)

            B_n_ = B(x)
            x_c_ = x_center(x, v_, B_n_)
            B_c_ = B(x_c_)

            lhs = Phi_pm(B_n_, B_c_, h, -1) * v_

            term_1 = Phi_pm(B_n, B_c, h, +1) * v
            term_2 = h / 2 * Psi_pm(B_n_, B_c_, h, -1) * E(x)
            term_3 = h / 2 * Psi_pm(B_n, B_c, h, +1) * E_n

            rhs = term_1 + term_2 + term_3

            return lhs - rhs
        end

        v = nlsolve(equation_twop, v, ftol=1e-16, xtol=1e-16).zero

    end
    return x_t, v_t
end
