using Revise # lets you change A2funcs without restarting julia!
include("A2_src.jl")
using Plots
using Statistics: mean, std
using Zygote
using Test
using Logging
using .A2funcs: log1pexp # log(1 + exp(x)) stable
using .A2funcs: factorized_gaussian_log_density
using .A2funcs: skillcontour!
using .A2funcs: plot_line_equal_skill!


function log_prior(zs)
  """
  Computes the log of the prior over all player's skills.
  """
  return factorized_gaussian_log_density(0, log.(1), zs)
end

function logp_a_beats_b(za,zb)
  """
  Computes the log-likelihood that player with skill za beat player with skill zb.
  """
  return -(log1pexp.(zb .- za))
end

function all_games_log_likelihood(zs,games)
  """
  Computes the log-likelihoods for those observed games.
  """
  #games = convert(Array{Int64}, games)
  zs_a = zs[games[:, 1], :]
  zs_b =  zs[games[:, 2], :]
  likelihoods = logp_a_beats_b(zs_a, zs_b)
  return sum(likelihoods, dims = 1)
end

function joint_log_density(zs,games)
  """
  Combines the log-prior and log-likelihood of the observations.
  """
  return prod.(log_prior(zs) .+ all_games_log_likelihood(zs, games))
end

@testset "Test shapes of batches for likelihoods" begin
  B = 15 # number of elements in batch
  N = 4 # Total Number of Players
  test_zs = randn(4,15)
  test_games = [1 2; 3 1; 4 2] # 1 beat 2, 3 beat 1, 4 beat 2
  @test size(test_zs) == (N,B)
  #batch of priors
  @test size(log_prior(test_zs)) == (1,B)
  # loglikelihood of p1 beat p2 for first sample in batch
  @test size(logp_a_beats_b(test_zs[1,1],test_zs[2,1])) == ()
  # loglikelihood of p1 beat p2 broadcasted over whole batch
  @test size(logp_a_beats_b.(test_zs[1,:],test_zs[2,:])) == (B,)
  # batch loglikelihood for evidence
  @test size(all_games_log_likelihood(test_zs,test_games)) == (1,B)
  # batch loglikelihood under joint of evidence and
  @test size(joint_log_density(test_zs,test_games)) == (1,B)
end


# Convenience function for producing toy games between two players.
two_player_toy_games(p1_wins, p2_wins) = vcat([repeat([1,2]',p1_wins), repeat([2,1]',p2_wins)]...)

# Example for how to use contour plotting code
plot(title="Example Gaussian Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
example_gaussian(zs) = exp(factorized_gaussian_log_density([-1.,2.],[0.,0.5],zs))
skillcontour!(example_gaussian)
plot_line_equal_skill!()
savefig(joinpath("plots","example_gaussian.pdf"))

# plot prior contours
plot(title="Prior Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )

log_prior_(zs) = exp(log_prior(zs))
skillcontour!(log_prior_)
plot_line_equal_skill!()
savefig(joinpath("plots","prior.jpg"))

# plot likelihood contours
plot(title="Likelihood Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )

logp_a_beats_b_(z_comb) = exp(logp_a_beats_b(z_comb[1], z_comb[2]))
skillcontour!(logp_a_beats_b_)
plot_line_equal_skill!()
savefig(joinpath("plots","likelihood"))

# plot joint contours with player A winning 1 game
plot(title="Example of A Winning 1 Game Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
game_outcome_1 = two_player_toy_games(1,0)
joint_log_density_(zs) = joint_log_density(zs, game_outcome_1)
game_1_0(zs) = exp.(joint_log_density_(zs))
skillcontour!(game_1_0)
plot_line_equal_skill!()
savefig(joinpath("plots","A_Wins_1_Game"))

# plot joint contours with player A winning 10 games
plot(title="Example of A Winning 10 Games Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
game_outcome_10 = two_player_toy_games(10,0)
game_10_0(zs) = exp(joint_log_density(zs, game_outcome_10))
skillcontour!(game_10_0)
plot_line_equal_skill!()
savefig(joinpath("plots","A_Wins_10_Games"))

# plot joint contours with player A winning 10 games and player B winning 10 games
plot(title="Example of Two Players Each Win 10 Games Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
game_outcome_10_10 = two_player_toy_games(10,10)
game_10_10(zs) = exp(joint_log_density(zs, game_outcome_10_10))
skillcontour!(game_10_10)
plot_line_equal_skill!()
savefig(joinpath("plots","Each_Wins_10_Games"))


function elbo(params,logp,num_samples)
  """
  Computes an unbiased estimate of the evidence lower bound.
  """
  samples = exp.(params[2]) .* randn(length(params[1]), num_samples) .+ params[1]
  #logp_estimate = factorized_gaussian_log_density(mean(logp(samples)), log.(std(logp(samples))), logp(samples)) # p(z, x)
  logp_estimate = logp(samples)
  logq_estimate = factorized_gaussian_log_density(params[1], params[2], samples) # q(z|x)
  return mean(logp_estimate - logq_estimate)
end

# Conveinence function for taking gradients
function neg_toy_elbo(params; games = two_player_toy_games(1,0), num_samples = 100)
  """
  Returns the -elbo estimate with num_samples many samples from q.
  """
  logp(zs) = joint_log_density(zs, games)
  return -elbo(params,logp, num_samples)
end


# Toy game
num_players_toy = 2
toy_mu = [-2.,3.] # Initial mu, can initialize randomly!
toy_ls = [0.5,0.] # Initial log_sigma, can initialize randomly!
toy_params_init = (toy_mu, toy_ls)


function fit_toy_variational_dist(init_params, toy_evidence; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  """
  An optimization function which takes initial variational parameters and the evidence, and returns am optimized paramaters by using gradient descent
  """
  params_cur = init_params
  for i in 1:num_itrs
    grad_params = gradient(params_cur -> neg_toy_elbo(params_cur; games = toy_evidence, num_samples = num_q_samples), params_cur)[1]
    params_cur =  params_cur .- lr .* grad_params
    @info "$(i)th loss: $(neg_toy_elbo(params_cur; games = toy_evidence, num_samples = num_q_samples))"
#    plot(title="Target Distribution and Variational Approximation Contour",
#    xlabel = "Player 1 Skill",
#    ylabel = "Player 2 Skill");
#    target_dist(params_cur) = exp(joint_log_density(params_cur, toy_evidence))
#    samples = exp.(params_cur[2]) .* randn(length(params_cur[1]), num_q_samples) .+ params_cur[1]
#    var_approx(samples) = exp(factorized_gaussian_log_density(params_cur[1], params_cur[2], samples))
#    skillcontour!(target_dist, colour=:red)
#    plot_line_equal_skill!()
#    display(skillcontour!(var_approx, colour=:blue))
#    savefig(joinpath("plots","TDvsVA3"))
  end
  return params_cur
end

# Fit q with SVI observing player A winning 1 game
param1 = fit_toy_variational_dist(toy_params_init, game_outcome_1; num_itrs=200, lr= 1e-2, num_q_samples = 10)

# Fit q with SVI observing player A winning 10 games
param10 = fit_toy_variational_dist(toy_params_init, game_outcome_10; num_itrs=200, lr= 1e-2, num_q_samples = 10)

# Fit q with SVI observing player A winning 10 games and player B winning 10 games
param10_10 = fit_toy_variational_dist(toy_params_init, game_outcome_10_10; num_itrs=200, lr= 1e-2, num_q_samples = 10)

## Question 4

# (a) Yes i, j, k are conditional independent

# (b)
# Load the Data
using MAT
vars = matread("tennis_data.mat")
player_names = vars["W"]
tennis_games = Int.(vars["G"])
num_players = length(player_names)
print("Loaded data for $num_players players")


function fit_variational_dist(init_params, tennis_games; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params
  for i in 1:num_itrs
    grad_params = gradient(params_cur -> neg_toy_elbo(params_cur; games = tennis_games, num_samples = num_q_samples), params_cur)[1]
    params_cur =  params_cur .- lr .* grad_params
    @info "loss: $(neg_toy_elbo(params_cur; games = tennis_games, num_samples = num_q_samples))" # report objective value with current parameters
  end
  return params_cur
end

# nitialize variational family
init_mu = randn(107) #random initialziation
init_log_sigma = rand(107) # random initialziation
init_params = (init_mu, init_log_sigma)

# Train variational distribution
trained_params = fit_variational_dist(init_params, tennis_games)


# (c) plot of the approx means and variances
μ_tennis = trained_params[1]
logstd = trained_params[2]
perm = sortperm(μ_tennis)

plot(μ_tennis[perm], yerror=exp.(logstd[perm]),
title="Approximate Mean and Variance of All Players",
xlabel = "Players",
ylabel = "Performance",
label = "Means")
savefig(joinpath("plots","PlayerSorted"))


# (d) 10 players with highest mean skill under variational model
top_ten = player_names[perm[98:107,]]


# (e) Joint posterior over "Roger-Federer" and ""Rafael-Nadal""
RF = findall(x -> x == "Roger-Federer", player_names)[1][1]
RN = findall(x -> x == "Rafael-Nadal", player_names)[1][1]

μ_RF = trained_params[1][RF]
μ_RN = trained_params[1][RN]
μ_FN = [μ_RF, μ_RN]

σ_RF = trained_params[2][RF]
σ_RN = trained_params[2][RN]
σ_FN = [σ_RF, σ_RN]

param_FN = (μ_FN, σ_FN)

plot(title="Joint Posterior of Roger Federer and Rafael Nadal",
xlabel = "Roger-Federer",
ylabel = "Rafael-Nadal",
legend=:bottomright)
samples_FN = exp.(param_FN[2]) .* randn(length(param_FN[1]), 10) .+ param_FN[1]
var_approx_FN(samples_FN) = exp(factorized_gaussian_log_density(param_FN[1], param_FN[2], samples_FN))
skillcontour!(var_approx_FN)
plot_line_equal_skill!()
savefig(joinpath("plots","FedererNadal"))

# (f)
lastmu = trained_params[1][75]
lastsig = trained_params[2][75]
mula = [μ_RF, lastmu]
sigla = [σ_RF, lastsig]
paramla = (mula, sigla)

# (g)
num_sample_MC = 10000
sample_MC_FN = exp.(param_FN[2]) .* randn(length(param_FN[1]), num_sample_MC) .+ param_FN[1]
sample_MC_RF = sample_MC_FN[1,:]
sample_MC_RN = sample_MC_FN[2,:]
prob_MC = length(findall(x -> x > 0, sample_MC_RF .- sample_MC_RN)) / num_sample_MC

#(h)
num_sample_MC = 10000
sample_MC_FN = exp.(param_FN[2]) .* randn(length(param_FN[1]), num_sample_MC) .+ param_FN[1]
sample_MC_RF = sample_MC_FN[1,:]
sample_MC_RN = sample_MC_FN[2,:]
prob_MC = length(findall(x -> x < 0, sample_MC_RF .- sample_MC_RN)) / num_sample_MC

# (i)
# (b), (c), (e)





using Distributions

ainit_mu = rand(Normal(10, 1), 107) #random initialziation
ainit_log_sigma = rand(107) # random initialziation
ainit_params = (ainit_mu, ainit_log_sigma)

# Train variational distribution
atrained_params = fit_variational_dist(ainit_params, tennis_games)


# (c) plot of the approx means and variances
aμ_tennis = atrained_params[1]
alogstd = atrained_params[2]
aperm = sortperm(aμ_tennis)

plot(aμ_tennis[aperm], yerror=exp.(alogstd[aperm]),
title="Approximate Mean and Variance of All Players",
xlabel = "Players",
ylabel = "Performance",
label = "Means")


# (d) 10 players with highest mean skill under variational model
atop_ten = player_names[aperm[98:107,]]


# (e) Joint posterior over "Roger-Federer" and ""Rafael-Nadal""
aRF = findall(x -> x == "Roger-Federer", player_names)[1][1]
aRN = findall(x -> x == "Rafael-Nadal", player_names)[1][1]

aμ_RF = atrained_params[1][aRF]
aμ_RN = atrained_params[1][aRN]
aμ_FN = [aμ_RF, aμ_RN]

aσ_RF = atrained_params[2][aRF]
aσ_RN = atrained_params[2][aRN]
aσ_FN = [aσ_RF, aσ_RN]

aparam_FN = (aμ_FN, aσ_FN)

plot(title="Joint Posterior of Roger Federer and Rafael Nadal",
xlabel = "Roger-Federer",
ylabel = "Rafael-Nadal",
legend=:bottomright)
asamples_FN = exp.(aparam_FN[2]) .* randn(length(aparam_FN[1]), 10) .+ aparam_FN[1]
avar_approx_FN(asamples_FN) = exp(factorized_gaussian_log_density(aparam_FN[1], aparam_FN[2], asamples_FN))
skillcontour!(avar_approx_FN)
plot_line_equal_skill!()


# (f)

# (g)
# CDF method
cdf(Normal(mean(μ_tennis), 1), 0)

# MC simulation
num_sample_MC = 10000
asample_MC_FN = exp.(aparam_FN[2]) .* randn(length(aparam_FN[1]), num_sample_MC) .+ param_FN[1]
asample_MC_RF = asample_MC_FN[1,:]
asample_MC_RN = asample_MC_FN[2,:]
aprob_MC = length(findall(x -> x > 0, asample_MC_RF .- asample_MC_RN)) / num_sample_MC

#(h)
# CDF method
1 - cdf(Normal(mean(μ_tennis), 1), 0)

# MC simulation
num_sample_MC = 10000
sample_MC_FN = exp.(param_FN[2]) .* randn(length(param_FN[1]), num_sample_MC) .+ param_FN[1]
sample_MC_RF = sample_MC_FN[1,:]
sample_MC_RN = sample_MC_FN[2,:]
prob_MC = length(findall(x -> x < 0, sample_MC_RF .- sample_MC_RN)) / num_sample_MC
