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
    return theta * x + (1. - theta) * x_c
end

function theta(x)
    # DONT TRUST SINC IN JULIA IT USES 2PI
    return ((x / 2.) / sin(x / 2.))^2
end

function parallel_velocity(x, v, epsilon)
    "Compute parallel velocity from position and velocity through Magnetic field"
    B_n_ = B(x, epsilon)
    b_n_ = norm(B_n_)

    return B_n_ / b_n_ * dot(B_n_ / b_n_, v)

end