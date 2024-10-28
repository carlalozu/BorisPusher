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


function B_hat(b)
    # 3x3 Matrix that gives the cross product with B
    return [0 -b[3] b[2]; b[3] 0 -b[1]; -b[2] b[1] 0]
end

function tanc(x)
    return x != 0 ? tanh(x / 2) / (x / 2) : 1.0
end


function Psi(h, B)
    b = norm(B)
    return I(3) + (1 - tanc(h*b/2))/(b^2) * B ^2
end

function x_center(x, v, B)
    # Center of the cyclotron motion
    return x + cross(v, B) / norm(B)^2
end

function x_bar(theta, x, x_c)
    # Average position
    return theta*x + (1 - theta)*x_c
end

function theta(h, B)
    return 1 / sinc(h*norm(B) / 2)^2
end

function Phi_1(h, X)
    x = norm(X)
    return I + (1-tanc(h*x/2))/x^2 * X^2
end

function Gamma(h, X)
    x = norm(X)
    return I - (1 - 1/sinc(h*x))/(h*x^2)*X^2
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