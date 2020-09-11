# 9/10/2020 © Shuowen Chen

# load packages
using Random, Distributions, LinearAlgebra, Statistics, FastGaussQuadrature
# set seed
Random.seed!(123)

# univariate tauchen
function tauchen(m, ρ, N, σ²)
    # INPUTS:
    # 1. m: multiple of unconditional standard deviation
    # 2. ρ: AR(1) coefficient
    # 3. N: number of grid points
    # 4. σ²: variance of the white noise
    # OUTPUTS:
    # 1. Transition probability matrix
    # 2. Sequence of grid points

    # unconditional standard deviation of the AR(1)
    std_un = sqrt(σ²/(1 - ρ^2));
    # max and min of the grid
    ymax = m*std_un;
    ymin = -ymax;
    # length of adjacent grid point
    w = (ymax - ymin)/(N - 1);
    # lay out the grids
    ygrid = collect(ymin:w:ymax);
    # Transition probability matrix (N by N)
    P = zeros(N, N);
    P[:, 1] = cdf.(Normal(0, 1), ((ymin .- ρ*ygrid .+ w/2)./sqrt(σ²)));
    P[:, end] = 1 .- cdf.(Normal(0, 1), ((ymax .- ρ*ygrid .- w/2)./sqrt(σ²)));
    for k = 2:(N - 1)
        P[:, k] = cdf.(Normal(0, 1), (ygrid[k] .- ρ*ygrid .+ w/2)./sqrt(σ²)) -
        cdf.(Normal(0, 1), (ygrid[k] .- ρ*ygrid .- w/2)./sqrt(σ²));
    end
    return(transition = P, grid = ygrid)
end

# Univariate Tauchen and Hussey (1991)
function tauchenhussey(ρ, μ, σ², N)
    # z_{t+1} = (1-rho)*mu + rho*z_{t} + epsilon_{t+1}
    # OUTPUTS:
    # 1. P: transition probability matrix
    # 2. grid: vector of grid points

    # A function from FastGaussQuadrature
    (x, w) = gausshermite(N);
    # Gaussian nodes and weights for normal distribution
    # Acknowledgement: I benefit from reading slides by Karen Kopecky
    # http://www.karenkopecky.net/Teaching/eco613614/Notes_DiscretizingAR1s.pdf
    grid = x .* sqrt(2*σ²) .+ μ;
    w = w ./ sqrt(pi);
    # Compute transition probability matrix
    P = zeros(N, N);
    for i = 1:N
        for j = 1:N
            EZprime = (1 - ρ)*μ + ρ*grid[i];
            P[i, j] = w[j] *
        end
    end
    # Normalization

end

m = 3;
rho = 0.5;
N = 9;
testing = tauchen(m, rho, N, 0.01);

# Solve for stationary distribution of the discretized sequence
p = I(9) - testing[1];
A = [p; ones(1, 9)];
B = [zeros(9); 1];
π = inv(A'*A)*(A'*B);

# Simulate from a Markov Chain
Pr_mat = testing[1];
grid = testing[2];
T = 500; # number of time periods
sim = 1000; # number of simulations

function simulationMC(Pr_mat, grid, T, sim)
    # INPUTS:
    # 1. Pr_mat: transition matrix
    # 2. grid: grid points
    # 3. T: number of observations in each simulation
    # 4. sim: number of simulations

    # OUTPUTS:
    # 1. Z: simulated time series

    # first obs of the time series
    initial_state = convert(Int, (N + 1)/2);
    # placeholder for simulated time series
    Z = zeros(T, sim);
    Z[1, :] .= grid[initial_state]; # need to broadcast
    # standard uniformly distributed rv for creating time series
    u = rand(Uniform(), (T-1, sim));
    # store ∑_{l=1}^{j}π_{i,l} for j = 1,...,N
    Pr_sum = cumsum(Pr_mat, dims = 2);
    old_state = initial_state;
    for i = 1:sim
        for t = 2:T
            new_state = findfirst(u[t-1, i] .<= Pr_sum[old_state, :])
            Z[t, i] = grid[new_state];
            old_state = new_state; # move to the next obs
        end
    end
    return(Z)
end

lala = simulationMC(Pr_mat, grid, T, sim)

# estimating AR(1) coefficient
hat = zeros(sim, 1);
for i = 1:sim
    hat[i] = cov(lala[1:end-1, i], lala[2:end, i])/var(lala[:, i]);
end
meanhat = mean(hat);
