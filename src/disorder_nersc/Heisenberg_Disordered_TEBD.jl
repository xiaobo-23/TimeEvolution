# 01/30/2025
# Run DMRG simulation to obtain the ground-state wave function of the 1D Heisenberg model with disordered interactions.
# Use time evolving block decimation (TEBD) to simulate the time evolution of the 1D J1-J2 Heisenberg model.    

using ITensors
using ITensorMPS
using LinearAlgebra
using MKL
using HDF5
using Random

include("Entanglement.jl")


# Because of the competition between BLAS and Strided.jl multithreading, we want to disable Strided.jl multithreading
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8  
OMP_NUM_THREADS = 8


# Monitor the number of threads used by BLAS and LAPACK
@show BLAS.get_config()
@show BLAS.get_num_threads()


# Define the parameters used in the simulation
const N = 200
const τ = 0.05
const ttotal = 100.0
const cutoff = 1E-10
const J1 = 1.0      # Antiferromagnetic coupling
const J2 = 0.0      # No next-nearest-neighbor interactions
const delta = 0.5   # No dimmerization


let 
    println("###########################################################################################################")
    println("###########################################################################################################")
    println("Start the time evolution of the 1D Heisenberg model with disorders.")
    println("The parameters used in the simulation are:")
    @show N, cutoff, τ, ttotal, J1, J2, delta   
    println("###########################################################################################################")
    println("###########################################################################################################")


    # Make an array of "site" indices
    s = siteinds("S=1/2", N; conserve_qns=false)

    # # Make gates (1, 2), (2, 3), ..., (N-1, N)
    # gates = ITensor[]
    # for index in 1 : N - 2
    #     s₁ = s[index]
    #     s₂ = s[index + 1]
    #     s₃ = s[index + 2]

    #     # Add two-site gate for nearest-neighbor interactions
    #     if index % 2 == 1
    #         hj = 1/2 * J1 * (1 + delta) * op("S+", s₁) * op("S-", s₂) + 1/2 * J1 * (1 + delta) * op("S-", s₁) * op("S+", s₂) + J1 * (1 + delta) * op("Sz", s₁) * op("Sz", s₂)
    #         Gj = exp(-im * τ/2 * hj)
    #     else
    #         hj = 1/2 * J1 * (1 - delta) * op("S+", s₁) * op("S-", s₂) + 1/2 * J1 * (1 - delta) * op("S-", s₁) * op("S+", s₂) + J1 * (1 - delta) * op("Sz", s₁) * op("Sz", s₂)
    #         Gj = exp(-im * τ/2 * hj)
    #     end
    #     push!(gates, Gj)


    #     # Add two-site gate for next-nearest-neighbor interactions
    #     hj_tmp = 1/2 * J2 * op("S+", s₁) * op("S-", s₃) + 1/2 * J2 * op("S-", s₁) * op("S+", s₃) + J2 * op("Sz", s₁) * op("Sz", s₃) 
    #     Gj_tmp = exp(-im * τ/2 * hj_tmp)    
    #     push!(gates, Gj_tmp)
    # end

    # # Add the last gate for the last two sites
    # s₁ = s[N - 1]
    # s₂ = s[N]
    # if (N - 1) % 2 == 1
    #     hj = 1/2 * J1 * (1 + delta) * op("S+", s₁) * op("S-", s₂) + 1/2 * J1 * (1 + delta) * op("S-", s₁) * op("S+", s₂) + J1 * (1 + delta) * op("Sz", s₁) * op("Sz", s₂)
    #     Gj = exp(-im * τ/2 * hj)
    # else
    #     hj = 1/2 * J1 * (1 - delta) * op("S+", s₁) * op("S-", s₂) + 1/2 * J1 * (1 - delta) * op("S-", s₁) * op("S+", s₂) + J1 * (1 - delta) * op("Sz", s₁) * op("Sz", s₂)
    #     Gj = exp(-im * τ/2 * hj)
    # end
    # push!(gates, Gj)
    
    # # Add reverse gates due to the the symmetric Trotter decomposition
    # append!(gates, reverse(gates))

    # Set up the random number generator to guarantee reproducibility   
    random_seed=0
    Random.seed!(random_seed * 10 + 123)

    # Run DMRG simulation to obtain the ground-state wave function
    os = OpSum()
    for index = 1 : N - 1
        effectiveJ = J1 * rand(Float64)
        # effectiveJ = J1
        @show index, effectiveJ
        # Construct the Hamiltonian for the Hewisenberg model with disorders.
        os += effectiveJ, "Sz", index, "Sz", index + 1
        os += 1/2 * effectiveJ, "S+", index, "S-", index + 1
        os += 1/2 * effectiveJ, "S-", index, "S+", index + 1
    end


    # os = OpSum()
    # for index = 1 : N - 1
    #     random_number = rand(Float64)
    #     if random_number < 0.5
    #         effectiveJ = J1 * (1 + delta)
    #     else
    #         effectiveJ = J1 * (1 - delta)
    #     end
        
    #     @show index, random_number, effectiveJ
    #     # Construct the Hamiltonian for the Hewisenberg model with disorders.
    #     os += effectiveJ, "Sz", index, "Sz", index + 1
    #     os += 1/2 * effectiveJ, "S+", index, "S-", index + 1
    #     os += 1/2 * effectiveJ, "S-", index, "S+", index + 1
    # end


    Hamiltonian = MPO(os, s)
    ψ₀ = MPS(s, n -> isodd(n) ? "Up" : "Dn")
    # ψ₀ = randomMPS(s, states; linkdims = 10)
    
    
    # Tune the parameters used in the DMRG simulation and run the simulation to obtain the ground-state wave function
    nsweeps = 20
    eigsolve_krylovdim = 25
    maxdim = [20, 50, 200, 2000]
    states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    E, ψ = dmrg(Hamiltonian, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim)
    
    
    # Measure physical observables including one-point, two-point functions, and entanglement entropy
    Sz₀ = expect(ψ, "Sz"; sites=1:N)
    Czz₀ = correlation_matrix(ψ, "Sz", "Sz"; sites=1:N)
   
    # Measure the von Neumann entanglement entropy on every single bond
    SvN = entanglement_entropy(ψ, N)
    @show SvN

    # Measure the bond dimension of the MPS
    chi = linkdims(ψ)
    @show chi
    # chi = Vector{Int}(undef, N - 1)
    # for index = 1 : N - 1
    #     chi[index] = dim(linkind(ψ, index))
    # end

    h5open("data/heisenberg_binomial_disorder_v$random_seed.h5", "w") do file
        write(file, "Psi", ψ)
        write(file, "Energy", E)
        write(file, "Sz T=0", Sz₀)
        write(file, "Czz T=0", Czz₀)
        write(file, "SvN", SvN)
        write(file, "Bond", chi)
        # write(file, "Sz Perturbed", Sz₁)
        # write(file, "Czz Perturbed", Czz₁)
        # write(file, "Czz", Czz)
        # write(file, "Czz Odd", Czz_odd) 
        # write(file, "Czz Even", Czz_even)
        # write(file, "Sz", Sz_all)
        # write(file, "Sz Odd", Sz_all_odd)
        # write(file, "Sz Even", Sz_all_even)
    end

    return
end