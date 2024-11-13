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