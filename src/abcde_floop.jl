#
# ### NOTE: this script is an addition to KissABC to implement
# ### multithreading support with in-place operations for the ABCDE method;
# ### for this we use FLoops and its @floop and @init macros
#
# # TODO/NOTE: how to really check if multithreading runs correctly?
# # maybe fix all random seeds
#
# # IMPORTANT NOTE: in this code we use multithreading in the form of "filling pre-allocated
# # output" (https://juliafolds.github.io/data-parallelism/tutorials/mutations/#filling_outputs)
# # this can be unsafe and cause data races! (dict, sparsearrays, Bit vectors, views)
# # but Arrays should be fine (and θs, logπ, Δs, nθs, nlogπ, nΔs are arrays)
# # println(typeof(θs)) => Vector{KissABC.Particle{Tuple{Float64, Float64, Float64}}}
# # println(typeof(logπ)) => Vector{Float64}
# # println(typeof(Δs)) => Vector{Float64}
#
# # NOTE: the op tuple operations (also on a Particle) seem to
# # be zero-allocating (both immutable types!, no heap memory needed) and can be
# # broadcasted (over multiple elements in a tuple or Particle.x); values inside
# # the tuples are all plain data (int, float), so that this should be no problem
# # for multithreading (although zero-allocation, nothing is changed in-place,
# # just plain data calculation in stack memory); in the end written to nθs
# # (which in each generation is created as deepcopy and creates allocations, by identity.());
# # see also seems tests in polylox_env_ABC_HSC_test.jl
# # so all things at θs[i], logπ[i], Δs[i] are immutable, such as Vector{SomeType}()
# # with SomeType immutable as in my question (https://discourse.julialang.org/t/how-to-implement-multi-threading-with-external-in-place-mutable-variables/62610/3)
#
# # NOTE: @reduce needed? if yes
# # use solution like this https://discourse.julialang.org/t/using-floops-jl-to-update-array-counters/58805
# # or this (histogram)? https://juliafolds.github.io/data-parallelism/tutorials/quick-introduction/
#
# function abcde_init!(prior, dist!, varexternal, θs, logπ, Δs, nparticles, rng, ex)
#     # calculate cost/dist for each particle
#     # (re-draw parameters if not finite)
#
#     @floop ex for i = 1:nparticles
#         # @init allows re-using mutable temporary objects within each base case/thread
#         # TODO/NOTE: in ThreadedEx mode one might observe a high % of garbage collection
#         # it was hard to check if this @init really works on my varexternal object...
#         # (compared with ve = deepcopy(varexternal) alone allocations were similar,
#         # but also in sequential mode, suggesting it is just not driving the allocations)
#         # NOTE: varexternal is the tuple wrapper of all mutatable external
#         # variables as cellstate, stats, ... (same in abcde_swarm!)
#         # NOTE: θs, logπ, Δs are read out but never written to, (for this
#         # nθs, nlogπ, nΔs are used), so this should be data race free
#         @init ve = deepcopy(varexternal)
#         trng=rng
#
#         # NOTE: I checked that Threads.threadid() also works in an floop
#         # NOTE: seems to have something with thread-safe random numbers, see
#         # https://discourse.julialang.org/t/multithreading-and-random-number-generators/49777/8
#         # NOTE/TODO: maybe this can be improved performance-wise (see FLoops docs
#         # on random numbers)
#         ex!=SequentialEx() && (trng=Random.default_rng(Threads.threadid());)
#
#         if isfinite(logπ[i])
#             Δs[i] = dist!(θs[i].x, ve)
#         end
#         while (!isfinite(Δs[i])) || (!isfinite(logπ[i]))
#             θs[i] = op(float, Particle(rand(trng, prior)))
#             logπ[i] = logpdf(prior, push_p(prior,θs[i].x))
#             Δs[i] = dist!(θs[i].x, ve)
#         end
#     end
# end
#
# function abcde_swarm!(prior, dist!, varexternal, θs, logπ, Δs, nθs, nlogπ, nΔs,
#                     ϵ_pop, ϵ_target, γ, nparticles, nsims, earlystop, rng, ex)
#     @floop ex for i in 1:nparticles
#         @init ve = deepcopy(varexternal)
#
#         if earlystop
#             Δs[i] <= ϵ_target && continue
#         end
#         trng=rng
#         ex!=SequentialEx() && (trng=Random.default_rng(Threads.threadid());)
#         s = i
#         ϵ = ifelse(Δs[i] <= ϵ_target, ϵ_target, ϵ_pop)
#         if Δs[i] > ϵ
#             # NOTE: Δs .<= Δs[i] is this data race safe? note that nΔs and Δs
#             # are not referenced through the identity.() broadcast
#             s=rand(trng,(1:nparticles)[Δs .<= Δs[i]])
#         end
#         a = s
#         while a == s
#             a = rand(trng,1:nparticles)
#         end
#         b = a
#         while b == a || b == s
#             b = rand(trng,1:nparticles)
#         end
#         # θp is a new Particle with new tuple values (.x) [see comment above]
#         θp = op(+,θs[s],op(*,op(-,θs[a],θs[b]), γ))
#         lπ = logpdf(prior, push_p(prior,θp.x))
#         w_prior = lπ - logπ[i]
#         log(rand(trng)) > min(0,w_prior) && continue
#         nsims[i]+=1
#         dp = dist!(θp.x, ve)
#         if dp <= max(ϵ, Δs[i])
#             nΔs[i] = dp
#             nθs[i] = θp
#             nlogπ[i] = lπ
#         end
#     end
# end
#
# function abcde!(prior, dist!, ϵ_target, varexternal; nparticles=50, generations=20, α=0, earlystop=false,
#                 verbose=true, rng=Random.GLOBAL_RNG, proposal_width=1.0,
#                 ex=ThreadedEx())
#     @info("Running on experimental abcde! method")
#     @info("Running abcde! with executor ", typeof(ex))
#
#     ### this seems to be initialisation
#     @assert 0<=α<1 "α must be in 0 <= α < 1."
#
#     # draw prior parameters for each particle
#     θs =[op(float, Particle(rand(rng, prior))) for i = 1:nparticles]
#     logπ = [logpdf(prior, push_p(prior,θs[i].x)) for i = 1:nparticles]
#
#     ve = deepcopy(varexternal)
#     Δs = fill(dist!(θs[1].x, ve),nparticles)
#
#     abcde_init!(prior, dist!, varexternal, θs, logπ, Δs, nparticles, rng, ex)
#     ###
#
#     ### actual ABC run
#     nsims = zeros(Int,nparticles)
#     γ = proposal_width*2.38/sqrt(2*length(prior))
#     iters=0
#     complete=1-sum(Δs.>ϵ_target)/nparticles
#     while iters<generations
#         iters+=1
#         # identity.() behaves like deepcopy(), i.e. == is true, === is false
#         # so there are n=new object that can be mutated without data races
#         nθs = identity.(θs)
#         nΔs = identity.(Δs)
#         nlogπ=identity.(logπ)
#
#         # returns minimal and maximal distance/cost
#         ϵ_l, ϵ_h = extrema(Δs)
#         if earlystop
#             ϵ_h<=ϵ_target && break
#         end
#         ϵ_pop = max(ϵ_target,ϵ_l + α * (ϵ_h - ϵ_l))
#
#         abcde_swarm!(prior, dist!, varexternal, θs, logπ, Δs, nθs, nlogπ, nΔs,
#                             ϵ_pop, ϵ_target, γ, nparticles, nsims, earlystop, rng, ex)
#
#         θs = nθs
#         Δs = nΔs
#         logπ = nlogπ
#         ncomplete = 1 - sum(Δs .> ϵ_target) / nparticles
#         if verbose && (ncomplete != complete || complete >= (nparticles - 1) / nparticles)
#             @info "Finished run:" completion=ncomplete nsim = sum(nsims) range_ϵ = extrema(Δs)
#         end
#         complete=ncomplete
#     end
#     conv=maximum(Δs) <= ϵ_target
#     if verbose
#         @info "End:" completion = complete converged = conv nsim = sum(nsims) range_ϵ = extrema(Δs)
#     end
#     θs = [push_p(prior, θs[i].x) for i = 1:nparticles]
#     l = length(prior)
#     P = map(x -> Particles(x), getindex.(θs, i) for i = 1:l)
#     length(P)==1 && (P=first(P))
#     (P=P, C=Particles(Δs), reached_ϵ=conv)
# end
#
# export abcde!
