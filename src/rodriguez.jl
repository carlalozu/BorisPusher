"""Matrix filter functions by means of the Rodriguez-like formula"""

using LinearAlgebra

function Psi(B::Vector, h::Float64)
    b = norm(B)
    return I + (1 - tan(h * b / 2) / (h * b / 2)) / b^2 * hat(B)^2
end

function Phi_1(B::Vector, h::Float64)
    b = norm(B)
    return I + (1 - (h * b) / sin(h * b)) / b^2 * hat(B)^2
end

function Phi_2(B::Vector, h::Float64)
    b = norm(B)
    return I + (1 - (h * b / 2)^2 / sin(h * b / 2)^2) / (b * b) * hat(B)^2
end

function Gamma(B::Vector, h::Float64)
    b = norm(B)
    return (1 - (h * b) / sin(h * b)) / (h * b^2) * hat(B)
end

function phi_1(B::Vector, h::Float64)
    b = norm(B)
    return I + (1 - cos(h * b)) / (h * b * b) * hat(B) + (1 - sin(h * b) / (h * b)) / (b * b) * hat(B)^2
end

function sinch(B::Vector, h::Float64)
    b = norm(B)
    return I + (1 - sin(h * b) / (h * b)) / (b * b) * hat(B)^2
end
