# Helper functions

function hat(X)
    """3x3 Matrix that gives the cross product with matrix X"""
    return [0 -X[3] X[2]; X[3] 0 -X[1]; -X[2] X[1] 0]
end

function x_center(x, v, B)
    """Center of the cyclotron motion"""
    return x + cross(v, B) / norm(B)^2
end

function x_bar(theta, x, x_c)
    """Point on the straight line connecting x_center and x"""
    return theta * x + (1.0 - theta) * x_c
end

function theta(x)
    """theta function"""
    # DONT TRUST SINC IN JULIA IT USES PI*X
    return ((x / 2.0) / sin(x / 2.0))^2
end

function parallel_velocity(x, v, epsilon)
    """Compute parallel velocity from position and velocity through Magnetic field"""
    B_n_ = B(x, epsilon)
    b_n_ = norm(B_n_)

    return B_n_ / b_n_ * dot(B_n_ / b_n_, v)
end

function Phi_pm_(B_bar, h, sign)
    """Compute Φⁿ₊ and Φⁿ₋ for implicit algorithm, one step map"""
    return phi_1(B_bar, -1 * sign * h)
end

function Psi_pm_(B, B_bar, h, sign)
    """Compute Ψⁿ₊ and Ψⁿ₋ for implicit algorithm, one step map"""
    Psi_n = Psi(B, h)
    Phi_n = Phi_pm_(B_bar, h, sign)
    Gamma_n = Gamma(B, h)
    return Psi_n + sign * 2.0 * Phi_n * Gamma_n
end

function Lambda(B, Bc, h)
    """Compute Λⁿ, for two point algorithm, one step map"""
    Phi1 = Phi_1(B, h)
    Phi2 = Phi_2(Bc, h)
    return inv(Phi2) * Phi1
end

function Phi_pm(B, Bc, h, sign)
    """Compute Φⁿ₊ and Φⁿ₋, for two point algorithm, one step map"""
    Lambda_n = Lambda(B, Bc, h)
    sinch_n = sinch(B, h)
    return (I - sign * 0.5 * h * Lambda_n * hat(B)) * sinch_n
end

function Psi_pm(B, Bc, h, sign)
    """Compute Ψⁿ₊ and Ψⁿ₋, for two point algorithm, one step map"""
    Psi_n = Psi(B, h)
    Phi_pm_n = Phi_pm(B, Bc, h, sign)
    Gamma_n = Gamma(B, h)
    return Psi_n + sign * 2.0 * Phi_pm_n * Gamma_n
end
