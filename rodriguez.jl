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
