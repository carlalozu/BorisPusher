function matrix_function(B, fun_)
    B1, B2, B3 = B
    b = norm(B)^2
  
    term1 = 0.5 * (2 * (b - 1 * B2^2 - 1 * B3^2) + (B2^2 + B3^2) * fun_(-1 * im*sqrt(b)) + (B2^2 + B3^2) * fun_(im*sqrt(b))) / b
    term2 = 0.5 * (2 * B1 * B2 - 1 * (B1 * B2 + im*sqrt(b) * B3) * fun_(-1 * im*sqrt(b)) + (-1 * B1 * B2 + im*sqrt(b) * B3) * fun_(im*sqrt(b))) / b
    term3 = 0.5 * (2 * B1 * B3 + (im*sqrt(b) * B2 - 1 * B1 * B3) * fun_(-1 * im*sqrt(b)) - 1 * (im*sqrt(b) * B2 + B1 * B3) * fun_(im*sqrt(b))) / b
  
    term4 = 0.5 * (2 * B1 * B2 + (-1 * B1 * B2 + im*sqrt(b) * B3) * fun_(-1 * im*sqrt(b)) - 1 * (B1 * B2 + im*sqrt(b) * B3) * fun_(im*sqrt(b))) / b
    term5 = 0.5 * (2 * B2^2 + (b - 1 * B2^2) * fun_(-1 * im*sqrt(b)) + (b - 1 * B2^2) * fun_(im*sqrt(b))) / b
    term6 = 0.5 * (2 * B2 * B3 - 1 * (im*sqrt(b) * B1 + B2 * B3) * fun_(-1 * im*sqrt(b)) + (im*sqrt(b) * B1 - 1 * B2 * B3) * fun_(im*sqrt(b))) / b
  
    term7 = 0.5 * (2 * B1 * B3 - 1 * (im*sqrt(b) * B2 + B1 * B3) * fun_(-1 * im*sqrt(b)) + (im*sqrt(b) * B2 - 1 * B1 * B3) * fun_(im*sqrt(b))) / b
    term8 = 0.5 * (2 * B2 * B3 + (im*sqrt(b) * B1 - 1 * B2 * B3) * fun_(-1 * im*sqrt(b)) - 1 * (im*sqrt(b) * B1 + B2 * B3) * fun_(im*sqrt(b))) / b
    term9 = 0.5 * (2 * B3^2 + (b - 1 * B3^2) * fun_(-1 * im*sqrt(b)) + (b - 1 * B3^2) * fun_(im*sqrt(b))) / b
  
    return real.([
        term1 term2 term3;
        term4 term5 term6;
        term7 term8 term9
    ])
  end


function tanc(x)
    return tan(x) / x
end


function sinch(x)
    return sinh(x) / x
end


# function Psi(h, B)
#     # Psi operator in Rodriguez-like formula
#     b = norm(B)
#     return I + (1 - tanc(h*b/2))/b^2 * hat(B)^2
# end

function Psi(B)
    function psi_(x)
        return tanh(x/2) / (x/2)
    end
    return matrix_function(B, psi_)
end

# function Phi_1(h, B)
#     # Phi_1 operator in Rodriguez-like formula
#     b = norm(B)
#     return I + (1-1/sinc(h*b))/b^2 * hat(B)^2
# end

function Phi_1(B)
    function phi_1_(x)
        return x/sinh(x)
    end
    return matrix_function(B, phi_1_)
end

# function Phi_2(h, B)
#     # Phi_1 operator in Rodriguez-like formula
#     b = norm(B)
#     return I + (1-1/sinc(h*b/2)^2)/b^2 * hat(B)^2
# end

function Phi_2(B)
    function phi_2_(x)
        return ((x/2) / sinh(x/2))^2
    end
    return matrix_function(B, phi_2_)
end

# function Gamma(h, B)
#     # Gamma operator in Rodriguez-like formula
#     b = norm(B)
#     return (1 - 1/sinc(h*b))/(h*b^2)*hat(B)
# end

function Gamma(B)
    function gamma_(x)
        return (x/sinh(x)-1)/x
    end
    return matrix_function(B, gamma_)
end

# function phi_1(h, B)
#     # phi_1 operator in Rodriguez-like formula
#     b = norm(B)
#     return (1 - cosc(h * b)) / (h * b^2) * hat(B)
# end

function phi_1(B)
    function phi1_(x)
        return (exp(x)-1)/x
    end
    return matrix_function(B, phi1_)
end

# Compute Ψⁿ₊ and Ψⁿ₋ for implicit algorithm
function Psi_pm_(hB, h_B_bar, sign)
    Psi_n = Psi(hB)
    Phi_n = phi_1(-1*sign*h_B_bar)
    Gamma_n = Gamma(hB)
    return Psi_n + sign * 2 * Phi_n * Gamma_n
end

# Compute Λⁿ
function Lambda(hB, hBc)
    Phi1 = Phi_1(hB)
    Phi2 = Phi_2(hBc)
    return inv(Phi2) * Phi1
end

# Compute Φⁿ₊ and Φⁿ₋
function Phi_pm(hB, hBc, sign)
    Lambda_n = Lambda(hB, hBc)
    sinch_n = matrix_function(hB, sinch)
    return (I - sign * 0.5 * Lambda_n * hat(hB)) * sinch_n  # Implementing ± correctly
end

# Compute Ψⁿ₊ and Ψⁿ₋ for two step map
function Psi_pm(hB, hBc, sign)
    Psi_n = Psi(hB)
    Phi_pm_n = Phi_pm(hB, hBc, sign)
    Gamma_n = Gamma(hB)
    return Psi_n + sign * 2 * Phi_pm_n * Gamma_n
end
