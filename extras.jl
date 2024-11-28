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

function boris_impA_2(x_0::Vector, v_0::Vector, t::Tuple, nt::Int, epsilon::Float64)
    """Implicit filtered Boris integrator"""
    
    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)
    
    # Arrays to store the state
    x_t = Array{Float64}(undef, 3, nt)
    v_t = Array{Float64}(undef, 3, nt)
    
    x = x_0
    v = v_0
    
    for i in 1:nt
        # Store the position and velocity
        x_t[:, i] = x # x^{n}
        v_t[:, i] = v # v^{n-1/2}
    
        E_n = E(x)
        B_n = B(x, epsilon)

        x_c = x_center(x, v, B_n)
        theta_n = theta(h*norm(B_n))
        x_bar_n = x_bar(theta_n, x, x_c)
        B_bar_n = B(x_bar_n, epsilon)


        # Full step of the position x^{n+1}
        x_ = x + h*Phi_pm_(h*B_bar_n, +1)*v + h^2/2*Psi_pm_(h*B_n, h*B_bar_n, +1)*E_n
    
        function equation_imp(v_)
            # At x+1 position
            B_n_ = B(x_, epsilon)
            x_c_ = x_center(x_, v_, B_n_)
            theta_n_ = theta(h*norm(B_n_))
            x_bar_n_ = x_bar(theta_n_, x_, x_c_)
            B_bar_n_ = B(x_bar_n_, epsilon)

            lhs = Phi_pm_(h*B_bar_n_, -1)*v_

            term_1 = Phi_pm_(h*B_bar_n, +1)*v
            term_2 = h/2*Psi_pm_(h*B_n_, h*B_bar_n_, -1)*E(x_)
            term_3 = h/2*Psi_pm_(h*B_n, h*B_bar_n, +1)*E(x)

            rhs = term_1 + term_2 + term_3
            return lhs - rhs
        end
        
        v_guess = deepcopy(v)
        v = nlsolve(equation_imp, v_guess).zero
        x = x_

    end
    return x_t, v_t
    end

function boris_twoPA_2(x_0::Vector, v_0::Vector, t::Tuple, nt::Int, epsilon::Float64)
    """Implicit filtered Boris integrator"""
    
    # Parameters
    (t0, tf) = t
    h = (tf - t0) / (nt - 1)
    
    # Arrays to store the state
    x_t = Array{Float64}(undef, 3, nt)
    v_t = Array{Float64}(undef, 3, nt)
    
    x = x_0
    v = v_0
    
    for i in 1:nt
        # Store the position and velocity
        x_t[:, i] = x # x^{n}
        v_t[:, i] = v # v^{n-1/2}
    
        E_n = E(x)
        B_n = B(x, epsilon)

        x_c = x_center(x, v, B_n)
        B_c = B(x_c, epsilon)

        # Full step of the position x^{n+1}
        x_ = x + h*Phi_pm(h*B_n, h*B_c, +1)*v + h^2/2*Psi_pm(h*B_n, h*B_c, +1)*E_n

        function equation_twop(v_)

            B_n_ = B(x_, epsilon)
            x_c_ = x_center(x_, v_, B_n_)
            B_c_ = B(x_c_, epsilon)

            lhs = Phi_pm(h*B_n_, h*B_c_, -1)*v_

            term_1 = Phi_pm(h*B_n, h*B_c, +1)*v
            term_2 = h/2*Psi_pm(h*B_n_, h*B_c_, -1)*E(x_) 
            term_3 = h/2*Psi_pm(h*B_n,  h*B_c, +1)*E_n
            
            rhs = term_1 + term_2 + term_3

            return lhs - rhs
        end
        
        v_guess = deepcopy(v)
        v = nlsolve(equation_twop, v_guess).zero
        x = x_

    end
    return x_t, v_t
    end
