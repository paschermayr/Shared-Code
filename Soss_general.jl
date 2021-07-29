###############################################################################
#Load packages and set default values
# I wrote all functions with the Distributions.jl package so far, because there were some distributions in MeasureTheory.jl that were not yet supported (Gamma, Dirichlet)
using Soss #Soss v0.20.0
using Distributions #Distributions v0.24.18
using BenchmarkTools #BenchmarkTools v1.1.1

#parameter used for models
hyperparam1 = 1.
hyperparam2 = 2.
μ           = 3.
σ           = 4.
p           = [.2, .8]
#data
T           = 100
data        = randn(T)
latent      = rand(1:2, T)

###############################################################################
#=
#Question 1 - Model Arguments
1.1 Normally, when I am using a PPL, I am used to write down the log posterior
as function of the data – this is also the main method to differentiate data
from parameter. As far as I understand, this is not the case for Soss?

1.2	From what I understand, only additional hyperpameter are arguments for a Soss Model?
If I would add, for instance, μ to hyperparam1 and hyperparam2, then the prior evaluation for mu would be
missing in the generated Soss code.
=#
MODELSOSS = Soss.@model (hyperparam1, hyperparam2) begin
#Model parameter
    μ ~ Distributions.Normal(hyperparam1, hyperparam2)
    σ ~ Distributions.Gamma()
# Data
    data ~ For(data) do ~
        Distributions.Normal( μ, σ)
    end
    return data
end
MODELSOSS2 = Soss.@model (hyperparam1, hyperparam2, μ) begin
#Model parameter
    μ ~ Distributions.Normal(hyperparam1, hyperparam2)
    σ ~ Distributions.Gamma()
# Data
    data ~ For(data) do ~
        Distributions.Normal(μ, σ)
    end
    return data
end

################################################################################
#=
#Question 2 - Various
2.1	How does Soss get the difference between argument/parameter and data:
ConditionalModel given
    arguments    (:hyperparam1, :hyperparam2, :μ, :σ)
    observations ()  # --> Always empty
2.2 What is the reason that data is not defined as function input of the model?
=#
modsoss1a = MODELSOSS( (hyperparam1=hyperparam1, hyperparam2=hyperparam2, μ = μ, σ = σ) )
modsossab = MODELSOSS( (hyperparam1=hyperparam1, hyperparam2=hyperparam2, μ = μ, σ = σ, data=data) )

################################################################################
#=
#Question 3 - Time series models
#3.1 Can I merge both loops in the model below, or is this done by Soss automatically?
#3.2 Reason for For Loops being written not in the standard Julia syntax?
=#
LATENTMODELSOSS = Soss.@model  begin
#Model parameter
    μ ~ Distributions.Normal()
    σ ~ Distributions.Gamma()
    p ~ Distributions.Dirichlet(2,2)
#latent - For Loop 1
    latent ~ For(data) do ~
        Distributions.Categorical(p)
    end
#Observed - For Loop 2
    data ~ For(data) do ~
        Distributions.Normal(μ, σ)
    end
    return latent, data
end

################################################################################
#=
#Question 4 - Performance difference to manual function implementation
#4.1 Reason for the big performance difference against manually writing down the
log posterior below. Is there a way to make the performance similar?
=#
"Manual implementation for log posterior of Model defined above"
function loglik_LATENTMODELSOSS(μ, σ, p, latent, data)
    lprior  = 0.0
    llik    = 0.0
    lprior += Distributions.logpdf(Distributions.Normal(), μ)
    lprior += Distributions.logpdf(Distributions.Gamma(), σ)
    lprior += Distributions.logpdf(Distributions.Dirichlet(2,2), p)

    latent_distr = Distributions.Categorical(p)
    for t in eachindex(data)
        llik += Distributions.logpdf( latent_distr, latent[t])
        llik += Distributions.logpdf( Distributions.Normal(μ, σ), data[t])
    end
    return llik + lprior
end

modsoss3 = LATENTMODELSOSS( (μ = μ, σ = σ, p = p ) )
modsoss4 = LATENTMODELSOSS( (μ = μ, σ = σ, p = p, latent=latent, data=data ) )

logdensity( modsoss3( (latent=latent, data=data) ) )
logdensity( modsoss4 )
loglik_LATENTMODELSOSS(μ, σ, p, latent, data)

@btime logdensity( $modsoss3( $(latent=latent, data=data) ) ) #48.400 μs (870 allocations: 32.75 KiB)
@btime logdensity( $modsoss4 ) #48.400 μs (870 allocations: 32.75 KiB)
@btime loglik_LATENTMODELSOSS($μ, $σ, $p, $latent, $data) #1.950 μs (1 allocation: 96 bytes)

################################################################################
#=
#Question 5 - Recovering Prior distributions from a Soss model:
Most MCMC algorithms work in an unconstrained space and I usually work with vectors
to represent all parameter. I am using Bijectors.jl to inverse transform all parameter into
the original space. I have seen that you used the TransformVariables.jl package
to do this for other tasks, but I only need the prior information for the parameter transformation via Bijectors
and I already have that contained in the model, so for me Bijectors.jl was always a bit
easier to work with.

#5.1 How do I recover the prior information of a parameter if the prior are defined
as a Vector of distributions? I added an example where I cannot use the use
rand()/rand.() of the recovered prior. (I did manage to obtain the priors for non-vectorized priors though)
#5.2 I believe there is no other way around than using eval() to grab the prior
distributions? For non-hierarchical models, I can just do this once and save
it somewhere, but for hierarchical models that would decrease speed a lot as I would need to to
this every time parameter change.
=#
PRIORMODEL = Soss.@model  begin
#Model parameter
    μ ~ Distributions.Normal()
    σ .~ [Distributions.Gamma(), Distributions.Gamma()]
#Observed
    data ~ For(data) do ~
        Distributions.Normal(μ, σ)
    end
    return data
end
priormodel = PRIORMODEL( (μ = 1., σ = [2., 3.] ) )
μ_prior = eval( priormodel.model.dists.μ )
σ_prior = eval( priormodel.model.dists.σ )
#working
rand(μ_prior)
Distributions.logpdf(μ_prior, 1.)

#Working?
Distributions.logpdf(σ_prior, [2., 3.])
#not working
rand(σ_prior)
rand.(σ_prior)
rand(σ_prior[1])

################################################################################
#=
#Question 6 - Forward simulating a latent variable model
I have seen that one can forward simulate data with a given Soss Model.
#6.1 How would I forward simulate the latent variable model from below,
such that I recover both the latent and observed data?
#6.2 Similarly, how would I sample data given a particular latent trajector?
6.3 I have seen both rand() and simulate() been used in previous posts.
Is there a difference between those 2?
=#
LATENTMODELSOSS2 = Soss.@model  begin
#Model parameter
    μ ~ Distributions.Normal()
    σ ~ Distributions.Gamma()
    p ~ Distributions.Dirichlet(2,2)
#latent - For Loop 1
    latent ~ For(latent) do ~
        Distributions.Categorical(p)
    end
#Observed - For Loop 2
    data ~ For(data) do ~
        Distributions.Normal(μ, σ)
    end
    return latent, data
end

_modsoss = LATENTMODELSOSS2( (μ = μ, σ = σ, p = p ) )
logdensity( _modsoss( (data=data, latent=latent) ) )
#6.1 - not working
Soss.rand( _modsoss ) #latent and data should be returned
Soss.simulate(_modsoss, :data, :latent)
Soss.simulate(_modsoss, :data)
#6.2 - not working
Soss.rand( _modsoss( (latent=latent,) ) ) #data should be returned given latent input
Soss.rand( _modsoss( (latent=latent,) ), :data )
Soss.simulate( _modsoss( (latent=latent,) ) ) #data should be returned given latent input
Soss.simulate( _modsoss( (latent=latent,) ), :data )

################################################################################
#=
#Question 7 - Prediction
Is it possible with Soss to predict a new data point given the data and parameter?
In some times series models, data is dependent on a function of previous data points,
for instance a GARCH model below. This is difficult to describe, so I have an example below:

#7.1 Is it possible here that I can get hₜ² for each data point in this example?
#7.2 Is it possible to predict 1 new data point with Soss? This would need to have hₜ² be computed
incrementally for all data points, before the new data point is sampled.
=#
GARCH = Soss.@model (hₜ²_initial, ) begin
#Model parameter
    μ ~ Distributions.Normal()
    ω ~ Distributions.Uniform()
    α ~ Distributions.Gamma()
    β ~ Distributions.Gamma()
#Set initial variance
    hₜ² = hₜ²_initial
#data
    data ~ For( 2:size(data, 1) ) do t
        #Update variance hₜ²
        hₜ² = ω + α*hₜ² + β*(data[t-1] - μ)^2
        Distributions.Normal(μ, sqrt(hₜ²) )
    end
    return data
end
garch = GARCH( (hₜ²_initial = 1., μ = 0.0, ω = 0.1, α = .5, β = .1) )
logdensity( garch( (data=data, ) ) )
#not working:
Soss.predict(garch( (data=data, ) ) )

#7.1 Manually computing hₜ²
μ = 0.0
ω = 0.1
α = .5
β = .1
hₜ²_vec = zeros(Float64, size(data, 1) )
hₜ²_vec[1] = 1.
for t in 2:size(data, 1)
    hₜ²_vec[t] = ω + α*hₜ²_vec[t-1] + β*(data[t-1] - μ)^2
end
hₜ²_vec
#7.2 Prediction of new data point given all previous data points
rand( Distributions.Normal(μ, sqrt(hₜ²_vec[end]) ) )
#=
 using Plots
 histogram( rand( Distributions.Normal(μ, sqrt(hₜ²_vec[end]) ), 10^5 ) )
=#

################################################################################
#=
#Question 8 - Markov Chains
I have seen that one can define distributions of a vector via a For-loop.
#8.1 What can be done if the first element of this vector has a different distribution than all other elements?
An example would be a first-order Markov Chain, where we have an initial and transition distribution.
I created an example below, but was unable to initiate it as Soss model (because latent has a tilde sign twice I assume)
=#

MARKOVSOSS = Soss.@model begin
#Model parameter
    p2 .~ [Distributions.Dirichlet(2,2), Distributions.Dirichlet(2,2)]
# Initial distribution
#!NOTE: Cannot execute this line - if marked out, everything works
    latent[1] ~ Distributions.Categorical(2)
# Transition distribution
    latent ~ For( 2:size(latent,1) ) do t
        Distributions.Categorical( p2[ latent[t-1] ] )
    end
    return latent
end
p2 = [ [.1, .9], [.4, .6] ]
markovsoss = MARKOVSOSS( (p2 = p2,) )
Soss.logdensity( markovsoss( (latent=latent, ) ) )
