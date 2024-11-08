# Helper functions and problem definitions

function U(x)
    # potential energy, scalar
    return 1 / sqrt(x[1] .^ 2 + x[2] .^ 2)
end

function B(x, epsilon)
    # magnetic field
    return [-x[1], 0, x[3] + 1.0 / epsilon]
end

function E(x)
    # electric field: minus gradient of potential
    return [x[1], x[2], 0] / (x[1] .^ 2 + x[2] .^ 2)^(3 / 2)
end

function hat(X)
    # 3x3 Matrix that gives the cross product with matrix X
    return [0 -X[3] X[2]; X[3] 0 -X[1]; -X[2] X[1] 0]
end

function tanc(x)
    return x != 0 ? tan(x) / x : 1.0
end

function tanch(x)
    return x != 0 ? tanh(x) / x : 1.0
end

function sinch(x)
    return x != 0 ? sinh(x) / x : 1.0
end

function x_center(x, v, B)
    # Center of the cyclotron motion
    return x + cross(v, B) / norm(B)^2
end

function x_bar(theta, x, x_c)
    # Average position
    return theta*x + (1 - theta)*x_c
end

function theta(x)
    return 1 / sinc(x / 2)^2
end

function Psi(h, B)
    # Psi operator in Rodriguez-like formula
    b = norm(B)
    return I + (1 - tanc(h*b/2))/b^2 * hat(B)^2
end

function Phi_1(h, B)
    # Phi_1 operator in Rodriguez-like formula
    b = norm(B)
    return I + (1-1/sinc(h*b))/b^2 * hat(B)^2
end

function Phi_2(h, B)
    # Phi_1 operator in Rodriguez-like formula
    b = norm(B)
    return I + (1-1/sinc(h*b/2)^2)/b^2 * hat(B)^2
end

function Gamma(h, B)
    # Gamma operator in Rodriguez-like formula
    b = norm(B)
    return (1 - 1/sinc(h*b))/(h*b^2)*hat(B)
end

function phi_1(h, B)
    # Gamma operator in Rodriguez-like formula
    b = norm(B)
    return (1 - exp(h*b))/(h*b^2)*hat(B)
end

function system!(du, u, p, t)
    # System of equations
    x1, x1_prime, x2, x2_prime, x3, x3_prime = u
    epsilon = p[1]

    # Equations
    du[1] = x1_prime            # x1' = x1_prime
    du[2] = x1 / (x1^2 + x2^2)^(3 / 2) + x2_prime * (1.0 / epsilon + x3) # x1''
    du[3] = x2_prime            # x2' = x2_prime
    du[4] = x2 / (x1^2 + x2^2)^(3 / 2) - x3_prime * x1 - x1_prime * (1.0 / epsilon + x3) # x2''
    du[5] = x3_prime            # x3' = x3_prime
    du[6] = x2_prime * x1       # x3''
end