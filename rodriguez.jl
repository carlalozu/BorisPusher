function Psi(B, h)
    # Psi operator in Rodriguez-like formula
    b = norm(B)
    return I + (1 - tan(h * b / 2) / (h * b / 2)) / b^2 * hat(B)^2
end

function Phi_1(B, h)
    # Phi_1 operator in Rodriguez-like formula
    b = norm(B)
    return I + (1 - (h * b) / sin(h * b)) / b^2 * hat(B)^2
end


function Phi_2(B, h)
    # Gamma operator in Rodriguez-like formula
    b = norm(B)
    return I + (1 - (h * b / 2)^2 / sin(h * b / 2)^2) / (b * b) * hat(B)^2
end

function Gamma(B, h)
    # Gamma operator in Rodriguez-like formula
    b = norm(B)
    return (1 - (h * b) / sin(h * b)) / (h * b^2) * hat(B)
end

function phi_1(B, h)
    # Gamma operator in Rodriguez-like formula
    b = norm(B)
    return I + (1 - cos(h * b)) / (h * b * b) * hat(B) + (1 - sin(h * b) / (h * b)) / (b * b) * hat(B)^2
end

function sinch(B, h)
    # sinch in Rodriguez-like formula
    b = norm(B)
    return I + (1 - sin(h * b) / (h * b)) / (b * b) * hat(B)^2
end

function Phi_pm_(B_bar, h, sign)
    # Compute Φⁿ₊ and Φⁿ₋ for implicit algorithm, two step map
    return phi_1(B_bar, -1 * sign * h)
end

function Psi_pm_(B, B_bar, h, sign)
    # Compute Ψⁿ₊ and Ψⁿ₋ for implicit algorithm, two step map
    Psi_n = Psi(B, h)
    Phi_n = Phi_pm_(B_bar, h, sign)
    Gamma_n = Gamma(B, h)
    return Psi_n + sign * 2.0 * Phi_n * Gamma_n
end

function Lambda(B, Bc, h)
    # Compute Λⁿ
    Phi1 = Phi_1(B, h)
    Phi2 = Phi_2(Bc, h)
    return inv(Phi2) * Phi1
end

function Phi_pm(B, Bc, h, sign)
    # Compute Φⁿ₊ and Φⁿ₋
    Lambda_n = Lambda(B, Bc, h)
    sinch_n = sinch(B, h)
    return (I - sign * 0.5 * h * Lambda_n * hat(B)) * sinch_n
end

function Psi_pm(B, Bc, h, sign)
    # Compute Ψⁿ₊ and Ψⁿ₋
    Psi_n = Psi(B, h)
    Phi_pm_n = Phi_pm(B, Bc, h, sign)
    Gamma_n = Gamma(B, h)
    return Psi_n + sign * 2.0 * Phi_pm_n * Gamma_n
end
