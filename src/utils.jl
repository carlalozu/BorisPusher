"""Helper functions"""

using LinearAlgebra

function hat(X::Vector)
    """3x3 operator version of cross product with matrix X"""
    return [0 -X[3] X[2]; X[3] 0 -X[1]; -X[2] X[1] 0]
end

function x_center(x::Vector, v::Vector, B::Vector)
    """Guiding center approximation"""
    return x + cross(v, B) / norm(B)^2
end

function x_bar(theta::Float64, x::Vector, x_c::Vector)
    """Point on the straight line connecting x_center and x"""
    return theta * x + (1.0 - theta) * x_c
end

function theta(x::Float64)
    """theta function"""
    # DONT TRUST SINC IN JULIA IT USES PI*X
    if x == 0
        return 1.0
    end
    return ((x / 2.0) / sin(x / 2.0))^2
end

function parallel_velocity(x::Vector, v::Vector, B::Function)
    """Compute parallel velocity from position and velocity through magnetic field"""
    B_n_ = B(x)
    b_n_ = norm(B_n_)

    return B_n_ / b_n_ * dot(B_n_ / b_n_, v)
end

function Phi_pm_(B_bar::Vector, h::Float64, sign::Int)
    """Compute Φⁿ₊ and Φⁿ₋ for implicit algorithm"""
    return phi_1(B_bar, -1 * sign * h)
end

function Psi_pm_(B::Vector, B_bar::Vector, h::Float64, sign::Int)
    """Compute Ψⁿ₊ and Ψⁿ₋ for implicit algorithm"""
    Psi_n = Psi(B, h)
    Phi_n = Phi_pm_(B_bar, h, sign)
    Gamma_n = Gamma(B, h)
    return Psi_n + sign * 2.0 * Phi_n * Gamma_n
end

function Lambda(B::Vector, Bc::Vector, h::Float64)
    """Compute Λⁿ for two point algorithm"""
    Phi1 = Phi_1(B, h)
    Phi2 = Phi_2(Bc, h)
    return inv(Phi2) * Phi1
end

function Phi_pm(B::Vector, Bc::Vector, h::Float64, sign::Int)
    """Compute Φⁿ₊ and Φⁿ₋ for two point algorithm"""
    Lambda_n = Lambda(B, Bc, h)
    sinch_n = sinch(B, h)
    return (I - sign * 0.5 * h * Lambda_n * hat(B)) * sinch_n
end

function Psi_pm(B::Vector, Bc::Vector, h::Float64, sign::Int)
    """Compute Ψⁿ₊ and Ψⁿ₋ for two point algorithm"""
    Psi_n = Psi(B, h)
    Phi_pm_n = Phi_pm(B, Bc, h, sign)
    Gamma_n = Gamma(B, h)
    return Psi_n + sign * 2.0 * Phi_pm_n * Gamma_n
end
