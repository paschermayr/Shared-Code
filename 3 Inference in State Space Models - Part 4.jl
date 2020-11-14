using Plots, Distributions, Bijectors, Parameters
import Base: count
import QuantEcon: gth_solve
################################################################################
################################################################################
################################################################################
#OLD code necessary to run PMCMC algorithm
using Plots, Distributions, Bijectors, Parameters
import Base: count
import QuantEcon: gth_solve

function sampleHMM(evidence::Vector{<:Distribution}, transition::Vector{<:Distribution}, T::Int64)
        #Initialize states and observations
        state = zeros(Int64, T)
        observation = zeros(Float64, T)

        #Sample initial s from initial distribution
        state[1] = rand( 1:length(transition) ) #not further discussed here
        observation[1] = rand( evidence[ state[1] ] )

        #Loop over Time Index
        for time in 2:T
                state[time] = rand( transition[ state[time-1] ] )
                observation[time] = rand( evidence[ state[time] ] )
        end
        return state, observation
end

#Helper function
function logsumexp(arr::AbstractArray{T}) where {T <: Real}
    max_arr = maximum(arr)
    max_arr + log(sum(exp.(arr .- max_arr)))
end

mutable struct ParticleFilter
        initial         ::      Distributions.Distribution
        transition      ::      Vector{<:Distributions.Distribution}
        observations    ::      Vector{<:Distributions.Distribution}
end

#Create a function to run a particle filter
function propose(pf::ParticleFilter, evidence::AbstractArray{T}; Nparticles=100, threshold=75) where {T<:Real}
        #Initialize variables
        @unpack initial, transition, observations = pf #Model parameter
        ℓlik = 0.0 #Log likelihood

        ℓweights = zeros(Float64, Nparticles)#Weights that are used to approximate log likelihood
        ℓweightsₙₒᵣₘ = similar(ℓweights)

        #initialize particles and sample first time point via initial distribution
        particles = zeros(Int64, size(evidence, 1), Nparticles )
        particles[1,:] =  rand(initial, Nparticles)

        #loop through time
        for t in 2:size(evidence, 1)

        #propagate particles forward
        particles[t, :] .= rand.( transition[ particles[t-1, :] ] )

        #Get new weights - normalized weights depend on t-1. a) resampling happens -> all weights the same, b) no resampling -> particles-weights in same order
        ℓweights .= logpdf.( observations[ particles[t, :] ], evidence[t] )
        ℓweightsₙₒᵣₘ .= ℓweights .- logsumexp(ℓweights)

        #Calculated after weights, as only normalized weights are resampled, i.e., resampling at time t has no impact at loglik increment at time t
        ℓlik += logsumexp(ℓweights) #add incremental liklihood

        #reweight particles if resampling threshold achieved
        if exp( -logsumexp(2. * ℓweightsₙₒᵣₘ) ) <= threshold
                paths = rand( Categorical( exp.(ℓweightsₙₒᵣₘ) ), Nparticles )
                particles .= particles[:, paths] #Whole trajectory!
        end

        end

        #Draw 1 trajectory path at the end
        path = rand( Categorical( exp.(ℓweightsₙₒᵣₘ) ) )
        trajectory = particles[:, path] #to keep type

        return ℓlik, particles, trajectory
end

#Create a function so that we can obtain initial distribution from transition matrix
function get_transition_param(transitionᵥ::Vector{<:Categorical})
    return reduce(hcat, [transitionᵥ[iter].p for iter in eachindex(transitionᵥ)] )'
end

using Distributions, Bijectors, Parameters
import Base: count

#A struct that contains necessary infos about θ
mutable struct ParamInfo{T, D<:Union{Nothing, <:Distributions.Distribution} }
    value   :: T #Can be Real, Integer, or a Arrays composed of it
    prior   :: D#Corresponding Prior of value, needs to fulfill boundary constraints!
end
ParamInfo(value::T) where {T} = ParamInfo(value, nothing)

#Check the number of parameter in ParamInfo struct
function count(paraminfo::ParamInfo)
    return length(paraminfo.value)
end

#A summary struct that contains all relevant parameter
mutable struct HMMParameter
    μ   :: Vector{ParamInfo} #Observation Distribution Mean
    σ   :: Vector{ParamInfo} #Observation Distribution Variance
#    τ   :: Vector{ParamInfo} #Observation Distribution Variance
end

#Wrapper to transform θ into an unconstrained space
function transform(parameter::ParamInfo)
    return Bijectors.link(parameter.prior, parameter.value)
end
#Wrapper to transform θₜ back into the original space, and store it in parameter struct
function inverse_transform!(parameter::ParamInfo, θₜ::T) where {T}
    value = Bijectors.invlink(parameter.prior, θₜ)
    @pack! parameter = value
    return value
end

#The same two functions dispatched on the summary struct
function transform(parameter::HMMParameter)
    θₜ = Float64[]
    for field_name in fieldnames( typeof( parameter ) )
        sub_field = getfield(parameter, field_name )
        append!(θₜ, transform.(sub_field) )
    end
    return θₜ
end
function inverse_transform!(parameter::HMMParameter, θₜ::Vector{T}) where {T}
    counter = 1
    for field_name in fieldnames( typeof( parameter ) )
        sub_field = getfield(parameter, field_name )
        dim = sum( count.(sub_field) )
        inverse_transform!.( sub_field, θₜ[ counter:(-1 + counter + dim )] )
        counter += dim
    end
end

#function to calculate prior including Jacobian
function calculate_logπₜ(parameter::ParamInfo)
    return logpdf_with_trans(parameter.prior, parameter.value, true)
end

#Wrapper to calculate prior including jacobian on all parameter
function calculate_logπₜ(parameter::HMMParameter)
    πₜ = 0.0
    for field_name in fieldnames( typeof( parameter ) )
        sub_field = getfield(parameter, field_name )
        πₜ += sum( calculate_logπₜ.(sub_field) )
    end
    return πₜ
end

# Struct that contains necessary infos for a Metropolis MCMC step
mutable struct Metropolis{T<:Real}
    θₜ           ::  Vector{T}
    scaling     :: Float64
end

# Function to propose 1 Metropolis step
function propose(sampler::Metropolis)
    @unpack θₜ, scaling = sampler
    return rand( MvNormal(θₜ, scaling) )
end
################################################################################
################################################################################
################################################################################
#NEW functions for current post

################################################################################
#Support function to transfer parameter into distributions
function get_distribution(μᵥ::Vector{ParamInfo}, σᵥ::Vector{ParamInfo})
    return [Normal(μᵥ[iter].value, σᵥ[iter].value) for iter in eachindex(μᵥ)]
end
get_distribution(param::HMMParameter) = get_distribution(param.μ, param.σ)

#Support function to grab all parameter in HMMParameter container
function get_parameter(parameter::HMMParameter)
    θ = Float64[]
    for field_name in fieldnames( typeof( parameter ) )
        sub_field = getfield(parameter, field_name )
        append!(θ, [sub_field[iter].value for iter in eachindex(sub_field) ] )
    end
    return θ
end

################################################################################
#Generate data
T = 100
evidence_true =  [Normal(2., 1.), Normal(-2.,1.)]
transition_true = [ Categorical([0.85, 0.15]), Categorical([0.5, 0.5]) ]

s, e = sampleHMM(evidence_true, transition_true, T)

#Plot data
plot( layout=(2,1), label=false, margin=-2Plots.px)
plot!(e, ylabel="observed data", label=false, subplot=1, color="gold4")
plot!(s, yticks = (1:2), ylabel="latent state", label=false, subplot=2, color="black")

################################################################################
#Generate Initial parameter estimates - for univariate normal observations
μᵥ = [ParamInfo(3.0, Normal() ), ParamInfo(-3.0, Normal() )] #Initial value and Prior
σᵥ = [ParamInfo(1.5, Gamma(2,2) ), ParamInfo(2.0, Gamma(2,2) )] #Initial value and Prior
param_initial = HMMParameter(μᵥ, σᵥ)

#Generate initial state trajectory
evidence_current = get_distribution(param_initial)
transition_current = deepcopy(transition_true)
initial_current = Categorical( gth_solve( Matrix(get_transition_param(transition_current) ) ) )
pf = ParticleFilter(initial_current, transition_current, evidence_current)

################################################################################
#Run PMCMC sampler
#=
loglik_curent, _, s_current = propose(pf, e)
#Generate initial Metropolis sampler
θₜ_inital = transform(param_initial)
metropolissampler = Metropolis(θₜ_inital, 1/4)
pf = pf
parameter = param_initial

#mcmc = metropolissampler
observations = e
iterations = 10
iter = 1
scaling = 1/4
=#

function sampling(pf::ParticleFilter, parameter::HMMParameter,
                  observations::AbstractArray{T}; iterations = 5000, scaling = 1/4) where {T}
    ######################################## INITIALIZATION
    #initial trajectory and parameter container
    trace_trajectory =  zeros(Int64, size(observations, 1), iterations)
    trace_θ =  zeros(Float64, size( get_parameter(parameter), 1), iterations)
    #Initial initial prior, likelihood, parameter and trajectory
    logπₜ_current = calculate_logπₜ(parameter)
    θ_current = get_parameter(parameter)
    loglik_current, _, s_current = propose(pf, observations)
    #Initialize MCMC sampler
    θₜ_current = transform(parameter)
    mcmc = Metropolis(θₜ_current, scaling)

    ######################################## PROPAGATE THROUGH ITERATIONS
    for iter in Base.OneTo(iterations)

        #propose a trajectory given current parameter
        θₜ_proposed = propose(mcmc)
        #propose new parameter given current parameter - start by changing modelparameter in pf with current θₜ_proposed
        inverse_transform!(parameter, θₜ_proposed)
        pf.observations = get_distribution(parameter)
        logπₜ_proposed = calculate_logπₜ(parameter)
        loglik_proposed, _, s_proposed = propose(pf, observations)
        #Accept or reject proposed trajectory and parameter
        accept_ratio = min(1.0, exp( (loglik_proposed + logπₜ_proposed) - (loglik_current + logπₜ_current) ) )
        accept_maybe =  accept_ratio > rand()
        if accept_maybe
            θ_current = get_parameter(parameter)
            s_current = s_proposed
            logπₜ_current = logπₜ_proposed
            loglik_current = loglik_proposed
            #Set current θ_current as starting point for Metropolis algorithm
            mcmc.θₜ = θₜ_proposed  #transform(parameter)
        end
        #Store trajectory and parameter in trace
        trace_trajectory[:, iter] = s_current
        trace_θ[:, iter] = θ_current
    end
    ########################################
    return trace_trajectory, trace_θ
end

trace_s, trace_θ = sampling(pf, param_initial, e; iterations = 3000, scaling = 1/4)

#Check how well we did - trajectories
plot_latent = Plots.plot(layout=(1,1), legend=:topright)
Plots.plot!(s, ylabel="latent state", label=false, color="black")
Plots.plot!( round.( mean(trace_s; dims=2) ), color="gold4", label="Posterior nearest state")

#Check how well we did - θ
plot_θ = Plots.plot(layout=(2,1), legend=:topright)
Plots.hline!([2.,-2.], ylabel="μ", xlabel="", subplot=1, color="black", legend=false)
Plots.plot!( trace_θ[1:2,:]', color="gold4", label="Posterior draws", subplot=1, legend=false)

Plots.hline!([1.], ylabel="σ", xlabel="PMCMC Iterations", subplot=2, color="black", legend=false)
Plots.plot!( trace_θ[3:4,:]', color="gold4", label="Posterior draws", subplot=2, legend=false)

#Trace as a function
function plot_states(trace, states_true)
    plot_latent = Plots.plot(layout=(1,1), legend=:topright)#, label="Trajectory") #, margin = -15Plots.px)
    Plots.plot!(states_true, ylabel="latent state", label=false, color="black")
    Plots.plot!( round.( mean(trace; dims=2) ), color="gold4", label="Posterior nearest state")
    Plots.plot!(trace, color="lightgrey", lw=0.1, subplot=1)
    for iter in 1:size(trace, 2)
        Plots.plot!(trace[:,iter], color="lightgrey", lw=0.5, subplot=1, label=false )
    end
    return plot_latent
end
plot_states(trace_s, s)
