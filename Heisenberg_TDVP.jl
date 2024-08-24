## 08/23/2024
## 
using ITensors
using ITensors: MPO, OpSum, dmrg, inner, random_mps, siteinds
using ITensorTDVP: tdvp
using Observers: observer

function main()
    n = 200
    s = siteinds("S=1/2", n)
  
    function heisenberg(n; J1 = 1.0, J2 = 0.5, Δ = 0.2)
        os = OpSum()

        if !iszero(J1)
            for j in 1:2:(n - 1)
                # @show j
                os += J1 * (1 + Δ) / 2, "S+", j, "S-", j + 1
                os += J1 * (1 + Δ) / 2, "S-", j, "S+", j + 1
                os += J1 * (1 + Δ), "Sz", j, "Sz", j + 1
            end
            
            for j in 2:2:(n - 2)
                # @show j
                os += J1 * (1 - Δ) / 2, "S+", j, "S-", j + 1
                os += J1 * (1 - Δ) / 2, "S-", j, "S+", j + 1
                os += J1 * (1 - Δ), "Sz", j, "Sz", j + 1
            end
        end

        if !iszero(J2)
            for j in 1:(n - 2)
                os += J2 / 2, "S+", j, "S-", j + 2
                os += J2 / 2, "S-", j, "S+", j + 2
                os += J2, "Sz", j, "Sz", j + 2
            end
        end
        return os
    end
  
    J1 = 1.0 
    J2 = 0.5
    H = MPO(heisenberg(n; J1, J2), s)
    ψ = random_mps(s, "↑"; linkdims=10)
  
    @show inner(ψ', H, ψ) / inner(ψ, ψ)
  
    # Obtain the ground state using DMRG
    e0, ϕ0 = dmrg(H, ψ; nsweeps=10, maxdim=100, cutoff=1e-10)
    @show inner(ϕ0', H, ϕ0) / inner(ϕ0, ϕ0), e0

    # Apply perturbations into the center of the chain on odd and even sites
    center_even = div(n, 2)
    center_odd  = div(n, 2) - 1
    
    ϕ_odd = deepcopy(ϕ0)
    ϕ_even = deepcopy(ϕ0)
    
    local_op = op("Sz", s[center_even])
    ϕ_even = apply(local_op, ϕ_even; cutoff)  
    # normalize!(ϕ_even)
 
    local_operator_odd = op("Sz", s[center_odd])
    ϕ_odd = apply(local_operator_odd, ϕ_odd; cutoff)
    # normalize!(ϕ_odd)

    step(; sweep) = sweep
    current_time(; current_time) = current_time
    return_state(; state) = state
    measure_sz(; state) = expect(state, "Sz"; sites=1:n)
    obs = observer(
        "steps" => step, "times" => current_time, "states" => return_state, "sz" => measure_sz
    )

    # init = MPS(s, n -> isodd(n) ? "Up" : "Dn")
    state = tdvp(
        H, -1.0im, ϕ0; time_step=-0.1im, cutoff=1e-12, (step_observer!)=obs, outputlevel=1
    )

    println("\nResults")
    println("=======")
    for n in 1:length(obs.steps)
        print("step = ", obs.steps[n])
        print(", time = ", round(obs.times[n]; digits=3))
        # print(", |⟨ψⁿ|ψⁱ⟩| = ", round(abs(inner(obs.states[n], ϕ0)); digits=3))
        print(", |⟨ψⁿ|ψᶠ⟩| = ", round(abs(inner(obs.states[n], state)); digits=3))
        print(", ⟨Sᶻ⟩ = ", round(obs.sz[n]; digits=3))
        println()
    end
    
    # ϕ = tdvp(
    #   H,
    #   -10.0,
    #   ψ;
    #   time_step=-1.0,
    #   nsweeps=10,
    #   maxdim=100,
    #   cutoff=1e-10,
    #   normalize=true,
    #   reverse_step=false,
    #   outputlevel=1,
    # )
    # Sz = expect(ϕ, "Sz"; sites=1:n)
    # @show inner(ϕ', H, ϕ) / inner(ϕ, ϕ)

    ϕ = tdvp(
      H,
      -1.0im,
      ϕ0;
    #   time_step=-1.0,
      nsweeps=10,
      maxdim=100,
      cutoff=1e-10,
      normalize=true,
      reverse_step=false,
      outputlevel=1,
    )
    Sz = expect(ϕ, "Sz"; sites=1:n)
    @show inner(ϕ', H, ϕ) / inner(ϕ, ϕ)
    
   

    ϕ_odd_time = tdvp(
      H,
      -2.0im,
      ϕ_odd;
    #   time_step=-1.0,
      nsweeps=20,
      maxdim=100,
      cutoff=1e-10,
      normalize=true,
      reverse_step=false,
      outputlevel=1,
    )
    Sz_odd = expect(ϕ_odd_time, "Sz"; sites=1:n)
    @show inner(ϕ_odd_time', H, ϕ_odd_time) / inner(ϕ_odd_time, ϕ_odd_time)


    ϕ_even_time = tdvp(
      H,
      -2.0im,
      ϕ_even;
    #   time_step=-1.0,
      nsweeps=20,
      maxdim=100,
      cutoff=1e-10,
      normalize=true,
      reverse_step=false,
      outputlevel=1,
    )
    Sz_odd = expect(ϕ_even_time, "Sz"; sites=1:n)
    @show inner(ϕ_even_time', H, ϕ_even_time) / inner(ϕ_even_time, ϕ_even_time)


    e2, ϕ2 = dmrg(H, ψ; nsweeps=10, maxdim=100, cutoff=1e-10)
    @show inner(ϕ2', H, ϕ2) / inner(ϕ2, ϕ2), e2
  
    h5open("Data/Heisenberg_Dimerized_TDVP_N$(n)_Time$(ttotal)_J2$(J₂)_Delta$(δ).h5", "w") do file
        write(file, "Sz", Sz)
        write(file, "Sz_odd", Sz_odd)
        write(file, "Sz_even", Sz_even)
    end

    return nothing
  end
  
  main()