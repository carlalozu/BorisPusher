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

function x_center(x, v, B)
    # Center of the cyclotron motion
    return x + cross(v, B) / norm(B)^2
end

function x_bar(theta, x, x_c)
    # Average position
    return theta * x + (1 - theta) * x_c
end

function theta(x)
    # DONT TRUST SINC IN JULIA IT USES 2PI
    return ((x / 2) / sin(x / 2))^2
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

function parallel_velocity(x, v, epsilon)
    "Compute parallel velocity from position and velocity through Magnetic field"
    B_n = B(x, epsilon)
    b_n = norm(B_n)

    return B_n / b_n * dot(B_n / b_n, v)

end