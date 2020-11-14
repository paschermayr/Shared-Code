using Distributions, Bijectors, Parameters
import Base: count

#A container with necessary information about θ
mutable struct ParamInfo{T, D<:Union{Nothing, <:Distributions.Distribution} }
    value   :: T #Can be Real, Integer, or a Arrays composed of it
    prior   :: D #Corresponding prior of value, needs to fulfill boundary constraints!
end
ParamInfo(value::T) where {T} = ParamInfo(value, nothing)

#Check the number of parameter in ParamInfo struct
function count(paraminfo::ParamInfo)
    return length(paraminfo.value)
end

#A summary container for all relevant parameter
mutable struct HMMParameter
    μ   :: Vector{ParamInfo} #Observation Distribution Mean
    σ   :: Vector{ParamInfo} #Observation Distribution Variance
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

#The same two functions dispatched on the summary container
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

#function to calculate prior including Jacobian adjustment
function calculate_logπₜ(parameter::ParamInfo)
    return logpdf_with_trans(parameter.prior, parameter.value, true)
end

#Wrapper to calculate prior including jacobian adjustment on all parameter
function calculate_logπₜ(parameter::HMMParameter)
    πₜ = 0.0
    for field_name in fieldnames( typeof( parameter ) )
        sub_field = getfield(parameter, field_name )
        πₜ += sum( calculate_logπₜ.(sub_field) )
    end
    return πₜ
end

#Create a Vector of univariate Normal Model parameter  and assign a prior to each of them:
μᵥ = [ParamInfo(-2.0, Normal() ), ParamInfo(2.0, Normal() )] #Initial value and Prior
σᵥ = [ParamInfo(1.0, Gamma() ), ParamInfo(1.0, Gamma(2,2) )] #Initial value and Prior

hmmparam = HMMParameter(μᵥ, σᵥ)

#Transform parameter:
transform(hmmparam)

#Sample parameter from an unconstrained distribution, then inverse transform and plug into our container:
θₜ_proposed = randn(4)
inverse_transform!(hmmparam, θₜ_proposed)
hmmparam #Check that parameter in hmmparam fulfill boundary conditions

# Container with information for a Metropolis MCMC step
mutable struct Metropolis{T<:Real}
    θₜ           ::  Vector{T}
    scaling     :: Float64
end

# Function to propose a Metropolis step
function propose(sampler::Metropolis)
    @unpack θₜ, scaling = sampler
    return rand( MvNormal(θₜ, scaling) )
end

#Create an initial sampler, and propose new model parameter:
θₜ_inital = randn(4)
metropolissampler = Metropolis(θₜ_inital, 1/length(θₜ_inital))
propose(metropolissampler)


mu = [ParamInfo(-2.0, Normal() ), ParamInfo(2.0, Normal() )]
sigma = [ParamInfo(1.0, Gamma() ), ParamInfo(1.0, Gamma(2,2) )]
trans = [ParamInfo([.5, .5], Dirichlet(2,2) ), ParamInfo([.5, .5], Dirichlet(2,2) )]

hmmparam = HMMParameter(mu, sigma)#, trans)
parameter = deepcopy(hmmparam)

θₜ = [10., 11, .6541, 15.2]
inverse_transform!(parameter, θₜ )
parameter
transform(parameter)


Bijectors.invlink(Gamma(2,2), 10.)
sampler = Metropolis(randn(4), 1/4)
propose(sampler)



distr1 = LKJ(2, .5)
distr2 =
rand(distr)
Distributions.product_distribution
