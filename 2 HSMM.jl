using Distributions

function sampleHSMM(evidence::Vector{<:Distribution}, duration::Vector{<:Distribution}, transition::Matrix{Float64}, T::Int64)
        #Initialize states and observations
        state = zeros(Int64, T)
        state_length = zeros(Int64, T)
        observation = zeros(Float64, T)

        #Sample initial s from initial distribution
        state[1] = rand( 1:size(transition, 1) ) #not further discussed here
        state_length[1] = rand( duration[ state[1] ] ) #not further discussed here
        observation[1] = rand( evidence[ state[1] ] )

        #Loop over Time Index
        for time in 2:T
                if state_length[time-1] > 0
                        state[time] = state[time-1]
                        state_length[time] = state_length[time-1] - 1
                        observation[time] = rand( evidence[ state[time] ] )
                else
                        state[time] = rand( Categorical( transition[ state[time-1], :] ) )
                        state_length[time] = rand( duration[ state[time] ] )
                        observation[time] = rand( evidence[ state[time] ] )
                end
        end
        #Return output
        return state, state_length, observation
end


using Plots

T = 5000
evidence =  [Normal(0., .5), Normal(0.,1.), Normal(0.,2.)]
duration =  [NegativeBinomial(100., .2), NegativeBinomial(10., .05), NegativeBinomial(50.,0.5)]
transition = [0.0 0.5 0.5;
              0.8 0.0 0.2;
              0.8 0.2 0.0;]
state, state_length, observation = sampleHSMM(evidence, duration, transition, T)

plot( layout=(3,1), label=false, margin=-2Plots.px)
plot!(observation, ylabel="data", label=false, subplot=1, color="gold4")
plot!(state, yticks = (1:3), ylabel="state", label=false, subplot=2, color="black")
plot!(state_length, ylabel="duration", label=false, subplot=3, color="blue")
