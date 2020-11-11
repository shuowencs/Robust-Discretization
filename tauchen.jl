# 10/24/2020 © Shuowen Chen
# This script simulates CAPM data using Tauchen & Hussey (1991) method
# as in Wright (2003) and explores sensitivity of Euler equation estimation
# to number of discretization points.

# load packages
using Random, Distributions, LinearAlgebra, Statistics, FastGaussQuadrature, Optim
# set seed
Random.seed!(123)

#======= Defining Functions =======#
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
    return(transition = P, grid = ygrid);
end

# Univariate Tauchen and Hussey (1991)
function tauchenhussey(ρ, a, σ², N)
    # PURPOSE:
    # Discretize the following AR(1) process:
    # z_{t+1} = a + ρ*z_{t} + epsilon_{t+1}
    # INPUTS:
    # 1. ρ: AR(1) coefficient
    # 2. a: intercept
    # 3. σ²: variance of epsilon
    # 4. N: number of grid points

    # OUTPUTS:
    # 1. P: transition probability matrix
    # 2. grid: vector of grid points

    # A function from FastGaussQuadrature
    (x, w) = gausshermite(N);
    # Gaussian nodes and weights for normal distribution
    # Acknowledgement: I benefit from reading slides by Karen Kopecky
    # http://www.karenkopecky.net/Teaching/eco613614/Notes_DiscretizingAR1s.pdf

    # unconditional mean of the process
    μ = a / (1 - ρ);
    # Establish the grid using Gaussian-Hermite node
    grid = x .* sqrt(2*σ²) .+ μ;
    # Rescale the Gaussian-Hermite weight
    w = w ./ sqrt(pi);
    # Compute transition probability matrix
    P = zeros(N, N);
    for i = 1:N
        for j = 1:N
            EZprime = a + ρ*grid[i];
            # Gauchen-Hussey weight
            P[i, j] = w[j] * pdf(Normal(EZprime, sqrt(σ²)), grid[j]) /
            pdf(Normal(μ, sqrt(σ²)), grid[j]);
        end
    end
    # Normalization
    for i = 1:N
        P[i, :] = P[i, :] ./ sum(P[i, :])
    end
    return(P, grid);
end

# An auxiliary function to be called in tauchenhusseyvar
function productrule(N, EZ, μ, Σ, grid_j, weight_j, k)
    # Purpose:
    # Compute individual transition prob from state i to j and aggregate
    # by Gauss product rule (tensor product)

    # Inputs:
    # 1. N: number of grid points
    # 2. EZ: expected value in state i (k by 1)
    # 3. μ: unconditional mean of VAR process (k by 1)
    # 4. Σ: var-cov matrix of VAR(1) process
    # 5. grid_j: grid points for the next state
    # 6. weight_j: Gauss-Hermite weight for the next state
    # 7. k: number of variables in the VAR

    # Outputs:
    # 1. prob: aggregate transition probability from state i to j

    den_ind = zeros(1, k);
    for i = 1:k
        den_ind[i] = weight_j/sqrt(pi) * pdf(Normal(EZ[i], diag(sqrt(Σ))[i]),
        grid_j[i]) / pdf(Normal(μ[i], diag(sqrt(Σ))[i]), grid_j[i])
    end
    den_agg = prod(den_ind);
    return(den_agg);
end

# Bivariate Tauchen-Hussey (1991)
function tauchenhusseyvar(k, A, B, Σ, N)
    # PURPOSE: Discretize the following VAR(1) process
    # Y_t = A + B * Y_t-1 + e_t,
    # where e_t ∼ N(0, Σ)
    # Acknowledgement: I benefit from reading the quadrature code by Tauchen,
    # which is available in his website.

    # INPUTS:
    # 1. k: number of variables in the VAR
    # 2. A: constant of VAR(1) (k by 1)
    # 3. B: VAR(1) coefficient matrix (k by k)
    # 4. Σ: iid var-cov matrix
    # 5. N: number of discrete points for each variable k in the VAR

    # OUTPUTS:
    # 1. P: transition matrix (Nᵏ by Nᵏ)
    # 2. Grid: grid points (k by Nᵏ).
    # Note: number of states in the system is equal to number of grids * number of lags,
    # since we consider VAR(1), number of states = number of grid points

    # unconditional mean of the process
    μ = inv(I(k) - B) * A;
    # obtain Gaussian-Hermite nodes and weights
    (x, w) = gausshermite(N);
    # Grid for each variable
    grid_ind = repeat(x, outer = [1, k])' .* sqrt.(2 * diag(Σ)) .+ μ;
    # total grid (k by N^k) and GH weights (N^k) using product rule
    grid = zeros(k, N^k);
    w_tot = zeros(k, N^k);
    for i = 1:k
        grid[i, :] = repeat(grid_ind[i, :], inner = [N^(k-i), 1],
                            outer = [N^(i-1), 1]);
        w_tot[i, :] = repeat(w, inner = [N^(k-i), 1], outer = [N^(i-1), 1]);
    end
    w_tot = prod(w_tot, dims = 1);
    # Compute transition probability matrix
    P = zeros(N^k, N^k);
    for i = 1:N^k
        for j = 1:N^k
            # expected value in state i at time t-1
            EZprime = A + B*grid[:, i]; # k by 1
            # Gauchen-Hussey
            P[i, j] = productrule(N, EZprime, μ, Σ, grid[:, j], w_tot[j], k);
        end
    end
    # Normalization
    for i = 1:N^k
        P[i, :] = P[i, :] ./ sum(P[i, :])
    end
    return(grid, P);
end

# Simulate data using Markov transition probability and grid points
function simulationMC(Pr_mat, grid, T, sim)
    # INPUTS:
    # 1. Pr_mat: transition matrix
    # 2. grid: grid points
    # 3. T: number of observations in each simulation
    # 4. sim: number of simulations
    # 5. k: number of variables

    # OUTPUT:
    # 1. Z: simulated states using MC (k by T for # sim times)
    # Economic meaning: log consumption and dividend growth data
    # 2. state: state index for stock return simulation

    # first obs of the MC (middle state)
    initial_state = convert(Int, floor((length(grid)/size(grid)[1]+1)/2));
    # placeholder for simulated states
    Z = zeros(k, T, sim);
    # initial state
    Z[:, 1, :] .= grid[:, initial_state]; # need to broadcast

    # placeholder for state index
    state = zeros(T, sim);
    # initial state
    state[1, :] .= initial_state;

    # This follows from Adda and Cooper (2003)
    # standard uniformly distributed rv for
    # determining the new state
    u = rand(Uniform(), (T-1, sim));
    # store ∑_{l=1}^{j}π_{i,l} for j = 1,...,N
    Pr_sum = cumsum(Pr_mat, dims = 2);
    old_state = initial_state;
    for i = 1:sim
        for t = 2:T
            new_state = findfirst(u[t-1, i] .<= Pr_sum[old_state, :])
            Z[:, t, i] = grid[:, new_state];
            # store the state index
            state[t, i] = new_state;
            # move to the next obs
            old_state = new_state;
        end
    end
    state = convert.(Int, state);
    return(Z, state);
end

# a function that directly simulates the VAR
function simulationVAR(T, sim, A, B, Σ, k)
    # INPUTS:
    # 1. T: number of observations in each simulation
    # 2. sim: number of simulations
    # 3. A: constant of VAR(1) (k by 1)
    # 4. B: VAR(1) coefficient matrix (k by k)
    # 5. Σ: iid var-cov matrix
    # 6. K: number of variables

    # OUTPUTS:
    # 1. TS: simulated time series (k by 1)

    # normal errors e_t ∼ N(0, Σ) for t = 2,...,T and each simulation
    err = zeros(k, T-1, sim);
    for i = 1:sim
        err[:, :, i] = rand(MvNormal([0; 0], Σ), T-1);
    end
    # placeholder for simulated VAR
    TS = zeros(k, T, sim);
    # initial observation (steady state)
    for i = 1:sim
        TS[:, 1, i] = inv(I(k) - B) * A;
    end
    # Then use the relationship Y_t = A + B*Y_t-1 + e_t
    for i = 1:sim
        for j = 2:T
            # Note the index for err is due to specification of err
            TS[:, j, i] = A + B*TS[:, j-1, i] + err[:, j-1, i];
        end
    end
    return(TS)
end

# Simulate stock return
function simulateReturn(β, γ, P, CG, DG, N, stateindex)
    # INPUTS:
    # 1. β: discount factor
    # 2. γ: CRRA preference
    # 3. P: probability transition matrix
    # 4. CG: consumption growth
    # 5. DG: dividend growth
    # 6. N: number of grid points for each variable

    # OUTPUTS:
    # 1. smreturn: simulated return data

    # Define transition matrix for return
    B = β*P;
    for j = 1:N^2
        B[:, j] = B[:, j] .* CG[j].^(-γ) .* DG[j];
    end
    # State contingent price-dividend ratio (N^2 by 1)
    psm = inv(I(N^2) - B) * B * ones(N^2);
    # Realized stock return from state i to j (N^2 by N^2)
    Rmat = zeros(N^2, N^2);
    for i = 1:N^2
        for j = 1:N^2
            Rmat[i, j] = DG[j]*(1+psm[j])/psm[i];
        end
    end
    # simulate the return data
    T = size(stateindex)[1] - 1;
    smreturn = zeros(T, 1);
    for t = 1:T
        smreturn[t] = Rmat[stateindex[t], stateindex[t+1]];
    end
    return(smreturn);
end

# A function to conduct 2 step GMM
function GMM2step(moments, gradients, k, initial, samplesize)
    # INPUTS:
    # 1. moments: an anonymous function that maps theta to moment conditions
    # 2. gradients: an anoymous function that mapes theta to gradient conditions
    # 3. k: number of moments
    # 4. initial: initial values for estimation (must be floating)
    # 5. samplesize: sample size of the moments
    # OUTPUTS:
    # 1. est: parameter estimate
    # 2. se: standard errors

    # sample average moments
    sample_m = theta -> vec(mean(moments(theta), dims = 1));
    # first step weighting matrix
    W = I(k);
    # objective function in the first step
    obj = theta -> sample_m(theta)'*W*sample_m(theta);
    # minimization using limited BFGS
    firststep = Optim.optimize(obj, initial, LBFGS());
    # second step weighting matrix
    W2 = inv(cov(moments(firststep.minimizer)));
    # objective function in the second step
    obj2 = theta -> sample_m(theta)'*W2*sample_m(theta);
    secondstep = Optim.optimize(obj2, initial, LBFGS());
    # compute the gradient
    # dimension of D: k (# of pars) by p (# of moments)
    D = reshape(mean(gradients(secondstep.minimizer), dims = 1), k, length(initial));
    var = inv(D'*inv(cov(moments(secondstep.minimizer)))*D);
    # compute standard errors
    se = sqrt.(diag(var)./samplesize);
    return(firststep.minimizer, secondstep.minimizer, se);
end

#======= Main Program Starts Here =======#
# Setting parameters
β = 1.139; # utility discount factor
γ = 13.7; # CRRA parameter
k = 2; # number of variables
A = [0.021; 0.04]; # Calibrated values as in Wright (2003)
B = [-0.161 0.017; 0.414 0.117];
Σ = [0.0012 0.00177; 0.00177 0.014];
N = 4; # number of grids for each variable
T = 400; # number of time periods
sim = 400; # number of simulations

# Generate grid points and transition probability
(grid, P) = tauchenhusseyvar(k, A, B, Σ, N);
# Simulate log consumption and dividend growth
(timeseries, stateindex) = simulationMC(P, grid, T, sim);

# Consumption and dividend growth
CG_tot = exp.(timeseries[1, :, :]);
DG_tot = exp.(timeseries[2, :, :]);
# use state index to generate return stock market return
R = zeros(T-1, sim);
for i = 1:sim
    R[:, i] = simulateReturn(β, γ, P, CG_tot[:, i], DG_tot[:, i], N, stateindex[:, i]);
end

# conduct estimation
container = zeros(sim, 2);
for i = 1:sim
    # intercept, lagged return and lagged consumption growth
    ins = [ones(T-2) R[1:end-1, i] CG_tot[1:end-2, i]];
    # sample size of the moments
    t_mom = size(instruments)[1];
    # Construct Euler equation
    euler = theta -> (theta[1] .* R[2:end, i] .* CG_tot[2:end-1, i] .^ (-theta[2]) .-1) .* ins;

    grad = theta -> hcat(R[2:end, i] .* CG_tot[2:end-1, i] .^ (-theta[2]) .* ins,
                             -(theta[1] .* R[2:end, i] .* CG_tot[2:end-1, i] .^ (-theta[2]) .* log.(CG_tot[2:end-1, i])) .* ins);

    gest = GMM2step(euler, grad, 3, [0.0, 13.0], t_mom);
    container[i,:] = gest[2];
end

mean(container, dims =1)
