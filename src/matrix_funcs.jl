"""Matrix function computation by means of matrix identities"""

using LinearAlgebra

function matrix_function(B::Vector, func::Function)
    b1, b2, b3 = B
    b = norm(B)

    term1 = 0.5 * (2 * func(0) * b1^2 + (b2^2 + b3^2) * func(-im * b) + (b2^2 + b3^2) * func(im * b)) / b^2
    term2 = 0.5 * (2 * func(0) * b1 * b2 - (b1 * b2 + im * b * b3) * func(-1 * im * b) + (-b1 * b2 + im * b * b3) * func(im * b)) / b^2
    term3 = 0.5 * (2 * func(0) * b1 * b3 + (im * b * b2 - 1 * b1 * b3) * func(-1 * im * b) - (im * b * b2 + b1 * b3) * func(im * b)) / b^2

    term4 = 0.5 * (2 * func(0) * b1 * b2 + (-b1 * b2 + im * b * b3) * func(-im * b) - (b1 * b2 + im * b * b3) * func(im * b)) / b^2
    term5 = 0.5 * (2 * func(0) * b2^2 + (b1^2 + b3^2) * func(-1 * im * b) + (b1^2 + b3^2) * func(im * b)) / b^2
    term6 = 0.5 * (2 * func(0) * b2 * b3 - (im * b * b1 + b2 * b3) * func(-im * b) + (im * b * b1 - b2 * b3) * func(im * b)) / b^2

    term7 = 0.5 * (2 * func(0) * b1 * b3 - (im * b * b2 + b1 * b3) * func(-im * b) + (im * b * b2 - b1 * b3) * func(im * b)) / b^2
    term8 = 0.5 * (2 * func(0) * b2 * b3 + (im * b * b1 - b2 * b3) * func(-im * b) - (im * b * b1 + b2 * b3) * func(im * b)) / b^2
    term9 = 0.5 * (2 * func(0) * b3^2 + (b1^2 + b2^2) * func(-im * b) + (b1^2 + b2^2) * func(im * b)) / b^2

    return real.([
        term1 term2 term3;
        term4 term5 term6;
        term7 term8 term9
    ])
end

function sinch(x::Float64)
    if x == 0
        return 1.0
    end
    return sinh(x) / x
end

function sinch(B::Vector, h::Float64)
    return matrix_function(h * B, sinch)
end

function Psi(B::Vector, h::Float64)
    function psi_(x::Float64)
        if x == 0
            return 1
        end
        return tanh(x / 2) / (x / 2)
    end
    return matrix_function(h * B, psi_)
end

function Phi_1(B::Vector, h::Float64)
    function phi_1_(x::Float64)
        return 1.0 / sinch(x)
    end
    return matrix_function(h * B, phi_1_)
end

function Phi_2(B::Vector, h::Float64)
    function phi_2_(x::Float64)
        if x == 0
            return 1.0
        end
        return (1 / sinch(x / 2))^2
    end
    return matrix_function(h * B, phi_2_)
end

function Gamma(B::Vector, h::Float64)
    function gamma_(x::Float64)
        if x == 0
            return 0.0
        end
        return (1 / sinch(x) - 1.0) / x
    end
    return matrix_function(h * B, gamma_)
end

function phi_1(B::Vector, h::Float64)
    function phi1_(x::Float64)
        if x == 0
            return 1.0
        end
        return (exp(x) - 1.0) / x
    end
    return matrix_function(h * B, phi1_)
end
