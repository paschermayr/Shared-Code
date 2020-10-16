using Plots, Distributions

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


T = 100
evidence =  [Normal(0., .5), Normal(0.,2.)]
transition = [ Categorical([0.7, 0.3]), Categorical([0.5, 0.5]) ]

state, observation = sampleHMM(evidence, transition, T)

plot( layout=(2,1), label=false, margin=-2Plots.px)
plot!(observation, ylabel="data", label=false, subplot=1, color="gold4")
plot!(state, yticks = (1:2), ylabel="state", xlabel="time", label=false, subplot=2, color="black")
