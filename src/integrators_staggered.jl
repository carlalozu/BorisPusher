using LinearAlgebra
using NLsolve

function boris(x_0::Vector, v_0::Vector, t::Tuple, nt::Int, B::Function, E::Function)
    """Standard Boris integrator"""
    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)

    # Arrays to store the statess
    x_t = Array{Float64}(undef, nt, 3)
    v_t = Array{Float64}(undef, nt, 3)

    v = v_0
    x = x_0

    # Initial half-step for velocity
    v = v - (cross(v, B(x)) + E(x)) * h / 2
    for n in 1:nt
        # Store the position at full step and velocity at half step
        x_t[n, :] = x # x^{n}
        v_t[n, :] = v # v^{n-1/2}

        E_n = E(x)
        B_n = B(x)

        # Half step of velocity due to electric field
        v_minus = v .+ h / 2 * E_n

        # Magnetic field rotation
        t_ = h / 2 * B_n
        s = 2 * t_ / (1 + dot(t_, t_))

        v_prime = v_minus .+ cross(v_minus, t_)
        v_plus = v_minus .+ cross(v_prime, s)

        # Half step of the velocity again
        v = v_plus .+ h / 2 * E_n

        # Full step of the position
        x = x .+ h * v
    end
    return x_t, v_t
end

function boris_expA(x_0::Vector, v_0::Vector, t::Tuple, nt::Int64, B::Function, E::Function)
    """Explicit filtered Boris integrator"""
    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)

    # Arrays to store the states
    x_t = Array{Float64}(undef, nt, 3)
    v_t = Array{Float64}(undef, nt, 3)

    v = v_0
    x = x_0

    # Initial half-step for velocity
    E_n = E(x)
    B_n = B(x)

    v = phi_1(B_n, h) * (v + h * Gamma(B_n, h) * E_n) - h / 2 * Psi(B_n, h) * E_n
    for n in 1:nt
        # Store the position at full step and velocity at half step
        x_t[n, :] = x # x^{n}
        v_t[n, :] = v # v^{n-1/2}

        E_n = E(x)
        B_n = B(x)

        v_plus = v + h / 2 * Psi(B_n, h) * E_n

        # for theta = 1, x_bar = x
        v_minus = exp(-h * hat(B_n)) * v_plus

        v = v_minus + h / 2 * Psi(B_n, h) * E_n

        # Full step of the position
        x = x + h * v

    end
    return x_t, v_t
end

function boris_impA(x_0::Vector, v_0::Vector, t::Tuple, nt::Int64, B::Function, E::Function)
    """Implicit filtered Boris integrator"""
    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)

    # Arrays to store the states
    x_t = Array{Float64}(undef, nt, 3)
    v_t = Array{Float64}(undef, nt, 3)

    v = v_0
    x = x_0

    # Initial half-step for velocity
    E_n = E(x)
    B_n = B(x)
    x_c = x_center(x, v, B_n)
    theta_n = theta(h * norm(B_n))
    x_bar_n = x_bar(theta_n, x, x_c)
    B_bar_n = B(x_bar_n)

    v = phi_1(B_bar_n, h) * (v + h * Gamma(B_n, h) * E_n) - h / 2 * Psi(B_n, h) * E_n
    for n in 1:nt
        # Store the position at full step and velocity at half step
        x_t[n, :] = x # x^{n}
        v_t[n, :] = v # v^{n-1/2}

        E_n = E(x)
        B_n = B(x)

        # v^{n-1/2}_{+}
        v_plus = v + h / 2 * Psi(B_n, h) * E_n

        theta_n = theta(h * norm(B_n))

        # solve for velocity
        function equation(v_n_)
            x_c = x_center(x, v_n_, B_n)
            x_bar_n = x_bar(theta_n, x, x_c)
            B_bar_n = B(x_bar_n)
            # v^{n-1/2}_{-} = exp(-h*hat_B_bar_n)) * v^{n-1/2}_{+}
            v_minus = exp(-h * hat(B_bar_n)) * v_plus

            # v^n
            lhs = v_n_
            rhs = 0.5 * Phi_1(B_bar_n, h) * (v_minus + v_plus) - h * Gamma(B_n, h) * E_n
            return lhs - rhs
        end

        # Solve numerically
        v_n = nlsolve(equation, v).zero

        # v^{n+1/2}
        x_c = x_center(x, v_n, B_n)
        x_bar_n = x_bar(theta_n, x, x_c)
        B_bar_n = B(x_bar_n)
        v_minus = exp(-h * hat(B_bar_n)) * v_plus
        v = v_minus + h / 2 * Psi(B_n, h) * E_n

        # Full step of the position x^{n+1}
        x = x + h * v

    end
    return x_t, v_t
end


function boris_twoPA(x_0::Vector, v_0::Vector, t::Tuple, nt::Int64, B::Function, E::Function)
    """Two-point filtered Boris integrator"""
    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)

    # Arrays to store the states
    x_t = Array{Float64}(undef, nt, 3)
    v_t = Array{Float64}(undef, nt, 3)

    v = v_0
    x = x_0

    B_n = B(x)
    E_n = E(x)

    x_c = x_center(x, v, B_n)
    B_c = B(x_c)

    # Initial half-step for velocity
    Phi_pm_n = Phi_pm(B_n, B_c, h, -1)  # Compute Φⁿ₊ or Φⁿ₋ based on the sign
    Psi_pm_n = Psi_pm(B_n, B_c, h, -1)  # Compute Ψⁿ₊ or Ψⁿ₋ based on the sign
    
    v = Phi_pm_n * v - h / 2 * Psi_pm_n * E_n
    for n in 1:nt
        # Store the position at full step and velocity at half step
        x_t[n, :] = x # x^{n}
        v_t[n, :] = v # v^{n-1/2}

        B_n = B(x)
        E_n = E(x)

        # v^{n-1/2}_{+}
        v_plus = v + h / 2 * Psi(B_n, h) * E_n

        # Approximate v^{n-1/2}_{-}
        v_minus = exp(-h * hat(B_n)) * v_plus
        tol = 1
        while tol > 1e-16
            # Compute v_n by (2.7), with B_n instead of B_bar_n
            v_n = Phi_1(B_n, h) * 1 / 2 * (v_minus + v_plus) - h * Gamma(B_n, h) * E_n
            x_c = x_center(x, v_n, B_n)
            B_c = B(x_c)

            # Express components of (7.1) in matrix form
            m_phi2 = Phi_2(B_c, h)
            m_phi1 = h / 2 * hat(B_n) * Phi_1(B_n, h)

            # Build the equation 
            M_LHS = m_phi2 + m_phi1
            v_RHS = (m_phi2 - m_phi1) * v_plus

            # solve for v_minus
            v_minus_ = inv(M_LHS) * v_RHS
            tol = norm(v_minus - v_minus_) / norm(v_minus_)

            v_minus = v_minus_
        end

        # Update the velocity
        v = v_minus + h / 2 * Psi(B_n, h) * E_n

        # Full step of the position x^{n+1}
        x = x + h * v

    end

    return x_t, v_t
end
