## 08/23/2024
## Develop time dependent variational principle (TDVP) to simulate the time evolution of spin models and compare with TEBD results 
## Designed to simulate neutron scattering experiments and quantum dynamics for the NEAT LDRD 


using ITensors
using ITensorMPS
using Observers: observer
using HDF5


# Set up the parameters used in the simulation
const n = 10                 # number of sites     
const ttotal = 10.0          # total time of evolution 
const J1 = 1.0               # nearest-neighbor interaction strength
const J2 = 0.5               # next-nearest-neighbor interaction strength
const Δ = 0.2                # dimerization parameter   


 # # Set up the Heisenberg Hamiltonian using OpSum 
  # function heisenberg(n; J1 = 1.0, J2 = 0.5, Δ = 0.2)
  #     os = OpSum()

  #     if !iszero(J1)
  #         for j in 1:2:(n - 1)
  #             # @show j
  #             os += J1 * (1 + Δ) / 2, "S+", j, "S-", j + 1
  #             os += J1 * (1 + Δ) / 2, "S-", j, "S+", j + 1
  #             os += J1 * (1 + Δ), "Sz", j, "Sz", j + 1
  #         end
          
  #         for j in 2:2:(n - 2)
  #             # @show j
  #             os += J1 * (1 - Δ) / 2, "S+", j, "S-", j + 1
  #             os += J1 * (1 - Δ) / 2, "S-", j, "S+", j + 1
  #             os += J1 * (1 - Δ), "Sz", j, "Sz", j + 1
  #         end
  #     end

  #     if !iszero(J2)
  #         for j in 1:(n - 2)
  #             os += J2 / 2, "S+", j, "S-", j + 2
  #             os += J2 / 2, "S-", j, "S+", j + 2
  #             os += J2, "Sz", j, "Sz", j + 2
  #         end
  #     end
  #     return os
  # end



function main()    
  # Define the spin sites in an MPS
  s = siteinds("S=1/2", n)

  # Set up the Heisenberg Hamiltonian as an MPO using OpSum
  os = OpSum()
  
  # Set up the nearest-neighbor interaction with dimerization 
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


  # Set up the next-nearest-neighbor interaction 
  if !iszero(J2)
      for j in 1:(n - 2)
          os += J2 / 2, "S+", j, "S-", j + 2
          os += J2 / 2, "S-", j, "S+", j + 2
          os += J2, "Sz", j, "Sz", j + 2
      end
  end

  
  # Convert OpSum to MPO
  H = MPO(os, s)
  ψ = random_mps(s, "↑"; linkdims=10)
  # @show inner(ψ', H, ψ) / inner(ψ, ψ)

  
  # Running DMRG to obtain the ground-state wave function and energy 
  println(repeat("#", 200))
  println("Running DMRG to obtain the ground-state wave function and energy...")
  e0, ϕ0 = dmrg(H, ψ; nsweeps=10, maxdim=200, cutoff=1e-10)
  Sz = expect(ϕ0, "Sz"; sites=1:n)
  @show e0 
  @show inner(ϕ0', H, ϕ0) / inner(ϕ0, ϕ0)
  @show Sz
  println(repeat("#", 200))
  println("")
  
  
  # Running TDVP to obtain the ground-state wave function and energy
  init = MPS(s, n -> isodd(n) ? "Up" : "Dn")
  println(repeat("#", 200))
  println("Running TDVP to time evolve the wave function...")
  step(; sweep) = sweep
  current_time(; current_time) = current_time
  return_state(; state) = state
  measure_sz(; state) = expect(state, "Sz"; sites = 1:n)
  obs = observer("steps" => step, "times" => current_time, "states" => return_state, "sz" => measure_sz)

  
  # Running TDVP time evolution in the imaginary time direction
  ϕ = tdvp(
    H,
    -20.0,
    ψ;
    nsteps=20,
    maxdim=100,
    cutoff=1e-10,
    normalize=true,
    outputlevel=1,
    nsite=2,
    (observer!)=obs,
  )
  
  
  @show inner(ϕ', H, ϕ) / inner(ϕ, ϕ)
  println(repeat("#", 200))
  println("")

  
  println("\nCompare Results")
  println(repeat("#", 200))
  for idx in 1:length(obs.steps)
    print("step = ", obs.steps[idx])
    print(", time = ", round(obs.times[idx]; digits=3))
    print(", |⟨ψⁿ|ψⁱ⟩| = ", round(abs(inner(obs.states[idx], ψ)); digits=3))
    print(", |⟨ψⁿ|ψᶠ⟩| = ", round(abs(inner(obs.states[idx], ϕ)); digits=3))
    # print(", ⟨Sᶻ⟩ = ", obs.sz[idx])
    println()
  end
  
  
  @show obs.sz[1]
  @show obs.sz[2]
  @show obs.sz[length(obs.sz)]
  @show Sz
  println(repeat("#", 200))
  println("")


  # # Apply perturbations into the center of the chain on odd and even sites
  # center_even = div(n, 2)
  # center_odd  = div(n, 2) - 1
  
  # ϕ_odd = deepcopy(ϕ0)
  # ϕ_even = deepcopy(ϕ0)
  
  # local_op = op("Sz", s[center_even])
  # ϕ_even = apply(local_op, ϕ_even; cutoff)  
  # # normalize!(ϕ_even)

  # local_operator_odd = op("Sz", s[center_odd])
  # ϕ_odd = apply(local_operator_odd, ϕ_odd; cutoff)
  # normalize!(ϕ_odd)


  # step(; sweep) = sweep
  # current_time(; current_time) = current_time
  # return_state(; state) = state
  # measure_sz(; state) = expect(state, "Sz"; sites=1:n)
  # obs = observer(
  #     "steps" => step, "times" => current_time, "states" => return_state, "sz" => measure_sz
  # )

  # # init = MPS(s, n -> isodd(n) ? "Up" : "Dn")
  # state = tdvp(
  #     H, -1.0im, ϕ0; time_step=-0.1im, cutoff=1e-12, (step_observer!)=obs, outputlevel=1
  # )

  # println("\nResults")
  # println("=======")
  # for n in 1:length(obs.steps)
  #     print("step = ", obs.steps[n])
  #     print(", time = ", round(obs.times[n]; digits=3))
  #     # print(", |⟨ψⁿ|ψⁱ⟩| = ", round(abs(inner(obs.states[n], ϕ0)); digits=3))
  #     print(", |⟨ψⁿ|ψᶠ⟩| = ", round(abs(inner(obs.states[n], state)); digits=3))
  #     print(", ⟨Sᶻ⟩ = ", round(obs.sz[n]; digits=3))
  #     println()
  # end
  
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

  
  
  # # Time evolve the wave function perturbed in the center of the chain: odd siteinds
  # ϕ_odd_time = tdvp(
  #   H,
  #   -ttotal,
  #   ϕ_odd;
  # #   time_step=-1.0,
  #   nsweeps=10,
  #   maxdim=100,
  #   cutoff=1e-10,
  #   normalize=true,
  #   reverse_step=false,
  #   outputlevel=1,
  # )
  # Sz_odd = expect(ϕ_odd_time, "Sz"; sites=1:n)
  # Czz_odd = correlation_matrix(ϕ_odd_time, "Sz", "Sz"; sites=1:n)
  # @show inner(ϕ_odd_time', H, ϕ_odd_time) / inner(ϕ_odd_time, ϕ_odd_time)


  # # Time evolve the wave fcuntion perturbed in the center of the chain: even site 
  # ϕ_even_time = tdvp(
  #   H,
  #   -ttotal,
  #   ϕ_even;
  # #   time_step=-1.0,
  #   nsweeps=10,
  #   maxdim=100,
  #   cutoff=1e-10,
  #   normalize=true,
  #   reverse_step=false,
  #   outputlevel=1,
  # )
  # Sz_even = expect(ϕ_even_time, "Sz"; sites=1:n)
  # Czz_even = correlation_matrix(ϕ_even_time, "Sz", "Sz"; sites=1:n)
  # @show inner(ϕ_even_time', H, ϕ_even_time) / inner(ϕ_even_time, ϕ_even_time)

  # e2, ϕ2 = dmrg(H, ψ; nsweeps=10, maxdim=100, cutoff=1e-10)
  # @show inner(ϕ2', H, ϕ2) / inner(ϕ2, ϕ2), e2


  # h5open("Data/Heisenberg_Dimerized_TDVP_N$(n)_Time$(imag(ttotal))_J2$(J2)_Delta$(Δ).h5", "w") do file
  #     write(file, "Sz", Sz)
  #     write(file, "Sz_odd", Sz_odd)
  #     write(file, "Sz_even", Sz_even)
  #     write(file, "Czz", Czz)
  #     write(file, "Czz_odd", Czz_odd)
  #     write(file, "Czz_even", Czz_even)
  # end

  return
end
  
main()