function matrix_function(B, fun_)
    B1, B2, B3 = B
    b = norm(B)

    term1 = 0.5 * (2 * fun_(0) * B1^2 + (B2^2 + B3^2) * fun_(-im * b) + (B2^2 + B3^2) * fun_(im * b)) / b^2
    term2 = 0.5 * (2 * fun_(0) * B1 * B2 - (B1 * B2 + im * b * B3) * fun_(-1 * im * b) + (-B1 * B2 + im * b * B3) * fun_(im * b)) / b^2
    term3 = 0.5 * (2 * fun_(0) * B1 * B3 + (im * b * B2 - 1 * B1 * B3) * fun_(-1 * im * b) - (im * b * B2 + B1 * B3) * fun_(im * b)) / b^2

    term4 = 0.5 * (2 * fun_(0) * B1 * B2 + (-B1 * B2 + im * b * B3) * fun_(-im * b) - (B1 * B2 + im * b * B3) * fun_(im * b)) / b^2
    term5 = 0.5 * (2 * fun_(0) * B2^2 + (B1^2 + B3^2) * fun_(-1 * im * b) + (B1^2 + B3^2) * fun_(im * b)) / b^2
    term6 = 0.5 * (2 * fun_(0) * B2 * B3 - (im * b * B1 + B2 * B3) * fun_(-im * b) + (im * b * B1 - B2 * B3) * fun_(im * b)) / b^2

    term7 = 0.5 * (2 * fun_(0) * B1 * B3 - (im * b * B2 + B1 * B3) * fun_(-im * b) + (im * b * B2 - B1 * B3) * fun_(im * b)) / b^2
    term8 = 0.5 * (2 * fun_(0) * B2 * B3 + (im * b * B1 - B2 * B3) * fun_(-im * b) - (im * b * B1 + B2 * B3) * fun_(im * b)) / b^2
    term9 = 0.5 * (2 * fun_(0) * B3^2 + (B1^2 + B2^2) * fun_(-im * b) + (B1^2 + B2^2) * fun_(im * b)) / b^2

    return real.([
        term1 term2 term3;
        term4 term5 term6;
        term7 term8 term9
    ])
end

function sinch(x)
    if x == 0
        return 1.0
    end
    return sinh(x) / x
end

function sinch(B, h)
    return matrix_function(h * B, sinch)
end

function Psi(B, h)
    function psi_(x)
        if x == 0
            return 1
        end
        return tanh(x / 2) / (x / 2)
    end
    return matrix_function(h * B, psi_)
end

function Phi_1(B, h)
    function phi_1_(x)
        return 1.0 / sinch(x)
    end
    return matrix_function(h * B, phi_1_)
end

function Phi_2(B, h)
    function phi_2_(x)
        if x == 0
            return 1.0
        end
        return (1 / sinch(x / 2))^2
    end
    return matrix_function(h * B, phi_2_)
end

function Gamma(B, h)
    function gamma_(x)
        if x == 0
            return 0.0
        end
        return (1 / sinch(x) - 1.0) / x
    end
    return matrix_function(h * B, gamma_)
end

function phi_1(B, h)
    function phi1_(x)
        if x == 0
            return 1.0
        end
        return (exp(x) - 1.0) / x
    end
    return matrix_function(h * B, phi1_)
end
