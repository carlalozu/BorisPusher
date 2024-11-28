function boris(x_0::Vector, v_0::Vector, t::Tuple, nt::Int, epsilon::Float64)
    """Standard Boris integrator"""
    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)

    # Arrays to store the states
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

        E_n =  E(x)
        B_n = B(x, epsilon)

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

function runge_kutta(x_0::Vector{Float64}, v_0::Vector{Float64}, t::Tuple{Float64, Float64}, nt::Int64, epsilon::Float64)
    """Runge-Kutta integrator"""

    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)
    time = t0:h:tf
    u0 = [x_0[1], v_0[1], x_0[2], v_0[2], x_0[3], v_0[3]]

    # Solving the system using a Runge-Kutta method
    prob = ODEProblem(system!, u0, (t0, tf), [epsilon])
    sol = solve(prob, Vern9(), saveat = time)

    # Recovering the states corresponding to the times
    u_t = sol(time).u

    # Extracting the positions and velocities
    x_t = [[u_t[i][1], u_t[i][3], u_t[i][5]] for i in 1:nt]
    x_t = hcat(x_t...)

    v_t = [[u_t[i][2], u_t[i][4], u_t[i][6]] for i in 1:nt]
    v_t = hcat(v_t...)

    return x_t, v_t
end

function boris_expA(x_0::Vector{Float64}, v_0::Vector{Float64}, t::Tuple{Float64, Float64}, nt::Int64, epsilon::Float64)
    """Explicit filtered Boris integrator"""

    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)

    # Arrays to store the state
    x_t = Array{Float64}(undef, 3, nt)
    v_t = Array{Float64}(undef, 3, nt)

    v = v_0
    x = x_0

    # Initial half-step for velocity
    E_n = E(x)
    B_n = B(x, epsilon)
    # v = v - (cross(v, B_n) + E_n)*h/2
    v = phi_1(h*B_n)*(v + h*Gamma(h*B_n)*E_n) - h/2* Psi(h*B_n)*E_n
    for i in 1:nt
        # Store the position and velocity
        x_t[:, i] = x
        v_t[:, i] = v

        E_n = E(x)
        B_n = B(x, epsilon)

        v_plus = v + h / 2 * Psi(h*B_n) * E_n

        # for theta = 1, x_bar = x
        v_minus = exp(-h*hat(B_n)) * v_plus

        v = v_minus + h / 2 * Psi(h*B_n) * E_n

        # Full step of the position
        x = x + h * v

    end
    return x_t, v_t
end

function boris_impA(x_0::Vector{Float64}, v_0::Vector{Float64}, t::Tuple{Float64, Float64}, nt::Int64, epsilon::Float64)
    """Implicit filtered Boris integrator"""
    
    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)
    
    # Arrays to store the state
    x_t = Array{Float64}(undef, 3, nt)
    v_t = Array{Float64}(undef, 3, nt)
    
    v = v_0
    x = x_0
    
    # Initial half-step for velocity
    E_n = E(x)
    B_n = B(x, epsilon)
    x_c = x_center(x, v, B_n)
    theta_n = theta(h*norm(B_n))
    x_bar_n = x_bar(theta_n, x, x_c)
    B_bar_n = B(x_bar_n, epsilon)
    
    # v = v - (cross(v, B(x, epsilon)) + E(x))*h/2
    v = phi_1(h*B_bar_n)*(v + h*Gamma(h*B_n)*E_n) - h/2* Psi(h*B_n)*E_n
    for i in 1:nt
        # Store the position and velocity
        x_t[:, i] = x # x^{n}
        v_t[:, i] = v # v^{n-1/2}
    
        E_n = E(x)
        B_n = B(x, epsilon)
    
        # v^{n-1/2}_{+}
        v_plus = v + h / 2 * Psi(h*B_n) * E_n
    
        theta_n = theta(h*norm(B_n))
        
        # solve for velocity
        function equation(v_n_)
            x_c = x_center(x, v_n_, B_n)
            x_bar_n = x_bar(theta_n, x, x_c)
            B_bar_n = B(x_bar_n, epsilon)
            # v^{n-1/2}_{-} = exp(-h*hat_B_bar_n)) * v^{n-1/2}_{+}
            v_minus = exp(-h*hat(B_bar_n)) * v_plus
    
            # v^n
            lhs = v_n_
            rhs = 0.5 * Phi_1(h*B_bar_n) * (v_minus + v_plus) - h * Gamma(h*B_n) * E_n
            return lhs - rhs
        end 

        # Solve numerically
        v_n = nlsolve(equation, v).zero

        # v^{n+1/2}
        x_c = x_center(x, v_n, B_n)
        x_bar_n = x_bar(theta_n, x, x_c)
        B_bar_n = B(x_bar_n, epsilon)
        v_minus = exp(-h*hat(B_bar_n)) * v_plus
        v = v_minus + h / 2 * Psi(h*B_n) * E_n
    
        # Full step of the position x^{n+1}
        x = x + h * v
    
    end
    return x_t, v_t
    end


function boris_twoPA(x_0::Vector{Float64}, v_0::Vector{Float64}, t::Tuple{Float64, Float64}, nt::Int64, epsilon::Float64)
    """Two-point filtered Boris integrator"""

    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)

    # Arrays to store the state
    x_t = Array{Float64}(undef, 3, nt)
    v_t = Array{Float64}(undef, 3, nt)

    v = v_0
    x = x_0

    B_n = B(x, epsilon)
    E_n = E(x)

    x_c = x_center(x, v, B_n)
    B_c = B(x_c, epsilon)

    # Initial half-step for velocity
    Phi_pm_n = Phi_pm(h*B_n, h*B_c, -1)  # Compute Φⁿ₊ or Φⁿ₋ based on the sign
    Psi_pm_n = Psi_pm(h*B_n, h*B_c, -1)  # Compute Ψⁿ₊ or Ψⁿ₋ based on the sign
    v = Phi_pm_n * v - h / 2 * Psi_pm_n * E_n
    for i in 1:nt
        # Store the position and velocity
        x_t[:, i] = x # x^{n}
        v_t[:, i] = v # v^{n-1/2}

        B_n = B(x, epsilon)
        E_n = E(x)

        # v^{n-1/2}_{+}
        v_plus = v + h / 2 * Psi(h*B_n) * E_n

        # Approximate v^{n-1/2}_{-}
        v_minus = exp(-h*hat(B_n)) * v_plus
        tol = 1
        while tol > 1e-16
            # Compute v_n by (2.7), with B_n instead of B_bar_n
            v_n = Phi_1(h*B_n) * 1/2 * (v_minus + v_plus) - h * Gamma(h*B_n) * E_n
            x_c = x_center(x, v_n, B_n)
            B_c = B(x_c, epsilon)

            # Express components of (7.1) in matrix form
            m_phi2 = Phi_2(h*B_c)
            m_phi1 = h/2 * hat(B_n) * Phi_1(h*B_n)

            # Build the equation 
            M_LHS = m_phi2 + m_phi1
            v_RHS = (m_phi2 - m_phi1) * v_plus

            # solve for v_minus
            v_minus_ = inv(M_LHS) * v_RHS
            tol = norm(v_minus - v_minus_) / norm(v_minus_)

            v_minus = v_minus_
        end

        # Update the velocity
        v = v_minus + h / 2 * Psi(h*B_n) * E_n

        # Full step of the position x^{n+1}
        x = x + h * v

    end

    return x_t, v_t
end
