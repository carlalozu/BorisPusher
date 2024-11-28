function matrix_function(B, fun_)
    B1, B2, B3 = B
    b = norm(B)
  
    term1 = 0.5 * (2 * fun_(0) * (b^2 - B2^2 - B3^2) + (B2^2 + B3^2) * fun_(-im*b) + (B2^2 + B3^2) * fun_(im*b)) / b^2
    term2 = 0.5 * (2 * fun_(0) * B1 * B2 - (B1 * B2 + im*b * B3) * fun_(-1 * im*b) + (-B1 * B2 + im*b * B3) * fun_(im*b)) / b^2
    term3 = 0.5 * (2 * fun_(0) * B1 * B3 + (im*b * B2 - 1 * B1 * B3) * fun_(-1 * im*b) - (im*b * B2 + B1 * B3) * fun_(im*b)) / b^2
  
    term4 = 0.5 * (2 * fun_(0) * B1 * B2 + (-B1 * B2 + im*b * B3) * fun_(-im*b) - (B1 * B2 + im*b * B3) * fun_(im*b)) / b^2
    term5 = 0.5 * (2 * fun_(0) * B2^2 + (b^2 - B2^2) * fun_(-1 * im*b) + (b^2 - B2^2) * fun_(im*b)) / b^2
    term6 = 0.5 * (2 * fun_(0) * B2 * B3 - (im*b * B1 + B2 * B3) * fun_(-im*b) + (im*b * B1 - B2 * B3) * fun_(im*b)) / b^2
  
    term7 = 0.5 * (2 * fun_(0) * B1 * B3 - (im*b * B2 + B1 * B3) * fun_(-im*b) + (im*b * B2 - B1 * B3) * fun_(im*b)) / b^2
    term8 = 0.5 * (2 * fun_(0) * B2 * B3 + (im*b * B1 - B2 * B3) * fun_(-im*b) - (im*b * B1 + B2 * B3) * fun_(im*b)) / b^2
    term9 = 0.5 * (2 * fun_(0) * B3^2 + (b^2 - B3^2) * fun_(-im*b) + (b^2 - B3^2) * fun_(im*b)) / b^2
  
    return real.([
        term1 term2 term3;
        term4 term5 term6;
        term7 term8 term9
    ])
  end


function tanc(x)
    if x==0
        return 1
    end
    return tan(x) / x
end


function sinch(x)
    if x==0
        return 1
    end
    return sinh(x) / x
end


# function Psi(h, B)
#     # Psi operator in Rodriguez-like formula
#     b = norm(B)
#     return I + (1 - tanc(h*b/2))/b^2 * hat(B)^2
# end

function Psi(B)
    function psi_(x)
        if x==0
            return 1
        end
        return tanh(x/2) / (x/2)
    end
    return matrix_function(B, psi_)
end

# function Phi_1(B, h)
#     # Phi_1 operator in Rodriguez-like formula
#     b = norm(B)
#     return I + (1-(h*b)/sin(h*b))/b^2 * hat(B)^2
# end

function Phi_1(B)
    function phi_1_(x)
        return 1/sinch(x)
    end
    return matrix_function(B, phi_1_)
end

# function Phi_2(B, h)
#     # Phi_1 operator in Rodriguez-like formula
#     b = norm(B)
#     return I + (1-1/sinc(h*b/2)^2)/b^2 * hat(B)^2
# end

function Phi_2(B)
    function phi_2_(x)
        if x==0
            return 1
        end
        return ((x/2) / sinh(x/2))^2
    end
    return matrix_function(B, phi_2_)
end

# function Gamma(B, h)
#     # Gamma operator in Rodriguez-like formula
#     b = norm(B)
#     return (1 - (h*b)/sin(h*b))/(h*b^2)*hat(B)
# end

function Gamma(B)
    function gamma_(x)
        if x==0
            return 0
        end
        return (x/sinh(x)-1)/x
    end
    return matrix_function(B, gamma_)
end


function phi_1(B)
    function phi1_(x)
        if x==0
            return 1
        end
        return (exp(x)-1)/x
    end
    return matrix_function(B, phi1_)
end

function Phi_pm_(h_B_bar, sign)
    # Compute Φⁿ₊ and Φⁿ₋ for implicit algorithm, two step map
    Phi_n = phi_1(-1*sign*h_B_bar)
    return Phi_n
end

function Psi_pm_(hB, h_B_bar, sign)
    # Compute Ψⁿ₊ and Ψⁿ₋ for implicit algorithm, two step map
    Psi_n = Psi(hB)
    Phi_n = Phi_pm_(h_B_bar, sign)
    Gamma_n = Gamma(hB)
    return Psi_n + sign * 2 * Phi_n * Gamma_n
end

function Lambda(hB, hBc)
    # Compute Λⁿ
    Phi1 = Phi_1(hB)
    Phi2 = Phi_2(hBc)
    return inv(Phi2) * Phi1
end

function Phi_pm(hB, hBc, sign)
    # Compute Φⁿ₊ and Φⁿ₋
    Lambda_n = Lambda(hB, hBc)
    sinch_n = matrix_function(hB, sinch)
    return (I - sign * 0.5 * Lambda_n * hat(hB)) * sinch_n 
end

function Psi_pm(hB, hBc, sign)
    # Compute Ψⁿ₊ and Ψⁿ₋
    Psi_n = Psi(hB)
    Phi_pm_n = Phi_pm(hB, hBc, sign)
    Gamma_n = Gamma(hB)
    return Psi_n + sign * 2 * Phi_pm_n * Gamma_n
end
