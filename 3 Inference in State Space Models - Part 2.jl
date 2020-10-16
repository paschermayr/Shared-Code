using Distributions, Parameters, Plots
import QuantEcon: gth_solve

mutable struct ParticleFilter
        initial         ::      Distributions.Distribution
        transition      ::      Vector{<:Distributions.Distribution}
        observations    ::      Vector{<:Distributions.Distribution}
end

#Helper function
function logsumexp(arr::AbstractArray{T}) where {T <: Real}
        max_arr = maximum(arr)
        max_arr + log(sum(exp.(arr .- max_arr)))
end
function logmeanexp(arr::AbstractVector{T}) where {T <: Real}
    log( 1/size(arr, 1) ) + logsumexp( arr )
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

        #Get new weights and calculate log likelihood
        ℓweights .= logpdf.( observations[ particles[t, :] ], evidence[t] )
        ℓweightsₙₒᵣₘ .= ℓweights .- logsumexp(ℓweights)
        ℓlik += logmeanexp(ℓweights) #add incremental likelihood

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

T = 100
HMMevidence =  [Normal(2., .5), Normal(-2.,2.)]
HMMtransition = [ Categorical([0.95, 0.05]), Categorical([0.5, 0.5]) ]

state, observation = sampleHMM(HMMevidence, HMMtransition, T)
plot( layout=(2,1), label=false, margin=-2Plots.px)
plot!(observation, ylabel="data", label=false, subplot=1, color="gold4")
plot!(state, yticks = (1:2), ylabel="state", label=false, subplot=2, color="black")

#Initialize PF
pf = ParticleFilter( Categorical( gth_solve( Matrix(get_transition_param(HMMtransition) ) ) ),
                    HMMtransition,
                    HMMevidence
                    )
ll, particles, trajectory = propose(pf, observation; Nparticles=500, threshold=500 )

ll, particles, trajectory = propose(pf, observation; Nparticles=1800, threshold=1800 )

Plots.plot(state, label="HMM latent states", xlabel="time", ylabel="latent state")
Plots.plot!( mean(particles; dims=2) , label="Particle Filter state trajectories")


#Check variance of likelihood estimate over a range of θ
function check_ll(pf::ParticleFilter, evidence::AbstractArray{T}, grid; runs = 20, Nparticles = 100) where {T<:Real}

        #Assign a matrix to store log likelihood estimate for each run
        ll_estimate = zeros(Float64, runs, length(grid))

        #Loop through the grid "runs" number of times, and assign the likelihood estimate to the preallocated matrix
        for iter in eachindex(grid)
                pf.observations[1] = Normal( grid[iter], pf.observations[1].σ)
                Base.Threads.@threads for idx in Base.OneTo(runs)
                        ll_estimate[idx, iter], _, _ = propose(pf, observation; Nparticles = Nparticles, threshold = Nparticles )
                end
        end

        #Return the log likelihood estimates
        return ll_estimate
end

grid = 0.0:0.05:5.0
ll_estimate = check_ll(pf, observation, grid; Nparticles = 500)
plot(grid, ll_estimate', seriestype = :scatter, ms=3.0, label="", xlabel="Parameter value", ylabel="log likelihood estimate")


ll_estimate = check_ll(pf, observation, grid; Nparticles = 50)
plot(grid, ll_estimate', seriestype = :scatter, ms=3.0, label="", xlabel="Parameter value", ylabel="log likelihood estimate")
