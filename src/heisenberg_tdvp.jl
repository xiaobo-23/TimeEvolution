## 08/23/2024
## Develop time dependent variational principle (TDVP) to simulate the time evolution of spin models and compare with TEBD results 
## Designed to simulate neutron scattering experiments and quantum dynamics for the NEAT LDRD 

using ITensors
using ITensorMPS
using Observers: observer
using HDF5
include("tdvp_models.jl")


# Set up the parameters used in the simulation
const n = 10                 # number of sites     
const ttotal = 10.0          # total time of evolution 
const J1 = 1.0               # nearest-neighbor interaction strength
const J2 = 0.5               # next-nearest-neighbor interaction strength
const Δ = 0.2                # dimerization parameter   


function main()    
  # Define the spin sites in an MPS
  s = siteinds("S=1/2", n)
  os = heisenberg_dimerized(n; J1=J1, J2=J2, Δ=Δ)

  
  # Convert OpSum to MPO
  H = MPO(os, s)
  ψ = random_mps(s, "↑"; linkdims=10)
  # @show inner(ψ', H, ψ) / inner(ψ, ψ)

  
  # Running DMRG to obtain the ground-state wave function and energy 
  println(repeat("#", 200))
  println("Running DMRG to obtain the ground-state wave function and energy...")
  e0, ϕ0 = dmrg(H, ψ; nsweeps=10, maxdim=200, cutoff=1e-10)
  Sz₀ = expect(ϕ0, "Sz"; sites=1:n)
  @show e0 
  @show inner(ϕ0', H, ϕ0) / inner(ϕ0, ϕ0)
  @show Sz₀
  println(repeat("#", 200))
  println("")
  

  # Save the ground-state wave function and energy into an HDF5 file as the initial state for TEBD time evolution
  output_filename = "data/heisenberg_input_n$(n).h5"
  h5open(output_filename, "w") do file
    write(file, "E0", e0)
    write(file, "Psi", ϕ0)
  end


  
  # Running TDVP to obtain the ground-state wave function and energy
  init = MPS(s, n -> isodd(n) ? "Up" : "Dn")
  
  # measure_os = [[("Id") for idx in 1:n] for idx in 1:n]
  # for i in 1:n
  #   measure_os[i][i] = "Sz"
  # end
  # @show measure_os
  # # measure_mpo = ops(measure_os, s)
  
  # measure_mpo = []
  # for idx in 1:n
  #   push!(measure_mpo, MPO(s, measure_os[idx]))
  # end
  # @show measure_mpo


  println(repeat("#", 200))
  println("Running TDVP to time evolve the wave function...")
  step(; sweep) = sweep
  current_time(; current_time) = current_time
  return_state(; state) = state
  measure_sz(; state) = expect(state, "Sz"; sites = 1:n)
  measure_czz(; state) = correlation_matrix(state, "Sz", "Sz"; sites = 1:n)
  # measure_unequal_sz(; state) = [inner(ψ', MPO(s, "Sz"), state) for idx in 1:n]
  # measure_unequal_sz(; state) = [inner(state', tmp_mpo, ψ) for tmp_mpo in measure_mpo]
  # obs = observer("steps" => step, "times" => current_time, "states" => return_state, "sz" => measure_sz, "sz_time" => measure_unequal_sz)
  obs = observer("steps" => step, "times" => current_time, "states" => return_state, "sz" => measure_sz, "czz" => measure_czz)
  

  # Running TDVP along the imaginary time direction to obtain the ground-state wave function
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

    
  println("\nCompare Results of the Imaginary-Time Evolution")
  println(repeat("#", 200))
  for idx in 1:length(obs.steps)
    println("step = ", obs.steps[idx])
    println(", time = ", round(obs.times[idx]; digits=3))
    println(", |⟨ψⁿ|ψⁱ⟩| = ", round(abs(inner(obs.states[idx], ψ)); digits=3))
    println(", |⟨ψⁿ|ψᶠ⟩| = ", round(abs(inner(obs.states[idx], ϕ)); digits=3))
    print(", ⟨Sᶻ⟩ = ", length(obs.sz[idx]))
    # print(", ⟨Sᶻ(t)Sᶻ(0)⟩ size ", length(obs.sz_time[idx]))
    # print(", ⟨Sᶻ(t)Sᶻ(0)⟩ = ", obs.sz_time[idx])
    println(", ⟨Czz⟩ size ", length(obs.czz[idx]))
    println()
  end
  
  # for idx in 1:length(obs.sz)
  #   print("step = ", idx)
  #   print(", ⟨Sᶻ⟩ = ", obs.sz[idx])
  #   println()
  # end

  println(repeat("#", 200))
  println("")
  

  # sz₁ = Matrix{Float64}(undef, length(obs.sz), n)
  # for idx in 1:length(obs.sz)
  #   sz₁[idx, :] = obs.sz[idx]
  # end
  # @show sz₁[1, :]
  # @show obs.sz[1]


  # czz₁ = Matrix{Float64}(undef, length(obs.sz), n*n)
  # for idx in 1:length(obs.sz)
  #   czz₁[idx, :] = obs.czz[idx]
  # end
  # @show czz₁[length(obs.czz), :]
  # @show obs.czz[length(obs.czz)]


  # # Save the TDVP results into an HDF5 file
  # h5open(output_filename, "cw") do file
  #     write(file, "steps", obs.steps)
  #     write(file, "time", round.(obs.times, digits=3))
  #     write(file, "sz", sz₁)
  #     write(file, "czz", czz₁)
  # end
  
  
  println(repeat("#", 200))
  println("Running TDVP to evolve the wave function in real time axis...")
  
  # Apply local perturbations to the ground-state wave function
  ϕ_copy = deepcopy(ϕ0)
  reference = div(n, 2)
  local_perturbation = op("Sz", s[reference])
  ϕ_copy = apply(local_perturbation, ϕ_copy; cutoff=1e-12)  
  normalize!(ϕ_copy)
  

  # Running TDVP along the imaginary time direction to obtain the ground-state wave function
  ϕ_final = tdvp(
    H,
    -10im,
    ψ;
    nsteps=50,
    maxdim=200,
    cutoff=1e-10,
    normalize=true,
    outputlevel=1,
    nsite=2,
    (observer!)=obs,
  )
  @show inner(ϕ_final', H, ϕ_final) / inner(ϕ_final, ϕ_final)
  println(repeat("#", 200))
  println("")

  
  println("\nCompare Results of the Real-Time Evolution")
  println(repeat("#", 200))
  for idx in 1:length(obs.steps)
    println("step = ", obs.steps[idx])
    println(", time = ", round(obs.times[idx]; digits=3))
    println(", |⟨ψⁿ|ψⁱ⟩| = ", round(abs(inner(obs.states[idx], ψ)); digits=3))
    println(", |⟨ψⁿ|ψᶠ⟩| = ", round(abs(inner(obs.states[idx], ϕ)); digits=3))
    print(", ⟨Sᶻ⟩ = ", obs.sz[idx])
    # print(", ⟨Sᶻ(t)Sᶻ(0)⟩ size ", length(obs.sz_time[idx]))
    # print(", ⟨Sᶻ(t)Sᶻ(0)⟩ = ", obs.sz_time[idx])
    # println(", ⟨Czz⟩ size ", length(obs.czz[idx]))
    println()
  end
  println(repeat("#", 200))
  println("")


  sz₂ = Matrix{Float64}(undef, length(obs.sz), n)
  for idx in 1:length(obs.sz)
    sz₂[idx, :] = obs.sz[idx]
  end
  @show sz₂[1, :]
  @show obs.sz[1]


  czz₂ = Matrix{ComplexF64}(undef, length(obs.sz), n*n)
  for idx in 1:length(obs.sz)
    czz₂[idx, :] = obs.czz[idx]
  end
  # @show czz₂[length(obs.czz), :]
  # @show obs.czz[length(obs.czz)]


  # Save the TDVP results into an HDF5 file
  h5open(output_filename, "cw") do file
      write(file, "steps", obs.steps)
      write(file, "time", round.(obs.times, digits=3))
      write(file, "sz", sz₂)
      write(file, "czz", czz₂)
  end

  return
end
  
main()