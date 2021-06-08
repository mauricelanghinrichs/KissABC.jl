
### NOTE: this script is an addition to KissABC to implement
### multithreading support with in-place operations for the ABCDE method;
### for this we use FLoops and its @floop and @init macros

# NOTE: @reduce needed? if yes
# use solution like this https://discourse.julialang.org/t/using-floops-jl-to-update-array-counters/58805
# or this (histogram)? https://juliafolds.github.io/data-parallelism/tutorials/quick-introduction/

function abcde_init!(prior, dist, θs, logπ, Δs, nparticles, rng, ex)
    # calculate cost/dist for each particle
    # (re-draw parameters if not finite)

    @floop ex for i = 1:nparticles
        # NOTE: in the beginning one can hopefully add the in-place
        # mutating constants for the dist! method via FLoops @init macro?
        # such as
        #@init c = deepcopy(cnst) # see examples in polylox_env_parallel.jl
        trng=rng

        # NOTE/TODO: check a bit better what this is supposed to do in parallel mode
        # at least I checked that Threads.threadid() also works in an floop
        ex!=SequentialEx() && (trng=Random.default_rng(Threads.threadid());)

        if isfinite(logπ[i])
            Δs[i] = dist(θs[i].x)
        end
        while (!isfinite(Δs[i])) || (!isfinite(logπ[i]))
            θs[i]=op(float, Particle(rand(trng, prior)))
            logπ[i] = logpdf(prior, push_p(prior,θs[i].x))
            Δs[i] = dist(θs[i].x)
        end
    end
end

function abcde_swarm!(prior, dist, θs, logπ, Δs, nθs, nlogπ, nΔs,
                    ϵ_pop, ϵ_target, γ, nparticles, nsims, earlystop, rng, ex)
    @floop ex for i in 1:nparticles
        # NOTE/TODO: here the mutated constants should go with @init
        # cellstate, stats, ... (same in abcde_init!)

        # NOTE: θs, logπ, Δs are read out but never written to, (for this
        # nθs, nlogπ, nΔs are used), so this should be data race free

        if earlystop
            Δs[i] <= ϵ_target && continue
        end
        trng=rng
        ex!=SequentialEx() && (trng=Random.default_rng(Threads.threadid());)
        s = i
        ϵ = ifelse(Δs[i] <= ϵ_target, ϵ_target, ϵ_pop)
        if Δs[i] > ϵ
            # NOTE: Δs .<= Δs[i] is this data race safe? note that nΔs and Δs
            # are not referenced through the identity.() broadcast
            s=rand(trng,(1:nparticles)[Δs .<= Δs[i]])
        end
        a = s
        while a == s
            a = rand(trng,1:nparticles)
        end
        b = a
        while b == a || b == s
            b = rand(trng,1:nparticles)
        end
        θp = op(+,θs[s],op(*,op(-,θs[a],θs[b]), γ))
        lπ = logpdf(prior, push_p(prior,θp.x))
        w_prior = lπ - logπ[i]
        log(rand(trng)) > min(0,w_prior) && continue
        nsims[i]+=1
        dp = dist(θp.x)
        if dp <= max(ϵ, Δs[i])
            nΔs[i] = dp
            nθs[i] = θp
            nlogπ[i] = lπ
        end
    end
end

function abcde!(prior, dist, ϵ_target; nparticles=50, generations=20, α=0, earlystop=false,
                verbose=true, rng=Random.GLOBAL_RNG, proposal_width=1.0,
                ex=ThreadedEx())
    @info("Running on experimental abcde! method")
    @info("Running abcde! with executor ", typeof(ex))

    ### this seems to be initialisation
    @assert 0<=α<1 "α must be in 0 <= α < 1."

    # draw prior parameters for each particle
    θs =[op(float, Particle(rand(rng, prior))) for i = 1:nparticles]
    logπ = [logpdf(prior, push_p(prior,θs[i].x)) for i = 1:nparticles]
    Δs = fill(dist(θs[1].x),nparticles)

    abcde_init!(prior, dist, θs, logπ, Δs, nparticles, rng, ex)
    ###

    ### actual ABC run
    nsims = zeros(Int,nparticles)
    γ = proposal_width*2.38/sqrt(2*length(prior))
    iters=0
    complete=1-sum(Δs.>ϵ_target)/nparticles
    while iters<generations
        iters+=1
        # identity.() behaves like deepcopy(), i.e. == is true, === is false
        # so there are n=new object that can be mutated without data races
        nθs = identity.(θs)
        nΔs = identity.(Δs)
        nlogπ=identity.(logπ)

        # returns minimal and maximal distance/cost
        ϵ_l, ϵ_h = extrema(Δs)
        if earlystop
            ϵ_h<=ϵ_target && break
        end
        ϵ_pop = max(ϵ_target,ϵ_l + α * (ϵ_h - ϵ_l))

        abcde_swarm!(prior, dist, θs, logπ, Δs, nθs, nlogπ, nΔs,
                            ϵ_pop, ϵ_target, γ, nparticles, nsims, earlystop, rng, ex)

        θs = nθs
        Δs = nΔs
        logπ = nlogπ
        ncomplete = 1 - sum(Δs .> ϵ_target) / nparticles
        if verbose && (ncomplete != complete || complete >= (nparticles - 1) / nparticles)
            @info "Finished run:" completion=ncomplete nsim = sum(nsims) range_ϵ = extrema(Δs)
        end
        complete=ncomplete
    end
    conv=maximum(Δs) <= ϵ_target
    if verbose
        @info "End:" completion = complete converged = conv nsim = sum(nsims) range_ϵ = extrema(Δs)
    end
    θs = [push_p(prior, θs[i].x) for i = 1:nparticles]
    l = length(prior)
    P = map(x -> Particles(x), getindex.(θs, i) for i = 1:l)
    length(P)==1 && (P=first(P))
    (P=P, C=Particles(Δs), reached_ϵ=conv)
end

export abcde!
