# 09/25/2025
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
const ttotal = 2.0
const cutoff = 1E-10
const J1 = 1.0       # Antiferromagnetic coupling
const J2 = 0.35      # No next-nearest-neighbor interactions
const delta = 0.04   # No dimmerization
const time_steps = Int(ttotal / τ)

let 
    println(repeat("#", 100))
    println(repeat("#", 100))
    println("Time evolve the J1-J2 Heisenberg model with disorders.")
    println("The parameters used in the simulation are:")
    @show N, cutoff, τ, ttotal, J1, J2, delta   
    println(repeat("#", 100))
    println(repeat("#", 100))


    # Set up the random number generator to guarantee reproducibility   
    # random_seed=23
    # Random.seed!(random_seed * 1000)

    
    #*************************************************************************************************************************
    #*************************************************************************************************************************
    # Make an array of "site" indices
    s = siteinds("S=1/2", N; conserve_qns=true)
    random_numbers = [rand(Float64) for _ in 1:N-1]

    
    # Set up nearest-neighbor interactions with disorders
    os = OpSum()
    bonds_with_disorders = 0
    for index in 1 : N - 1
        # random_number = rand(Float64)
        random_number = random_numbers[index]
        if random_number < 0.1
            normalizedJ = J1 * 0.6
            bonds_with_disorders += 1
            @show index, normalizedJ, random_number
        else
            normalizedJ = J1
        end
        # @show index, normalizedJ

        if isodd(index)
            effectiveJ = normalizedJ * (1 + delta)
        else
            effectiveJ = normalizedJ * (1 - delta)
        end

        os += effectiveJ, "Sz", index, "Sz", index + 1
        os += 1/2 * effectiveJ, "S+", index, "S-", index + 1
        os += 1/2 * effectiveJ, "S-", index, "S+", index + 1
    end 
    println("")
    println(repeat("#", 200))
    println("The number of bonds with disorders is: $bonds_with_disorders")
    println(repeat("#", 200))
    println("")


    # Set up next-nearest-neighbor interactions
    for index in 1 : N - 2
        os += J2, "Sz", index, "Sz", index + 2
        os += 1/2 * J2, "S+", index, "S-", index + 2
        os += 1/2 * J2, "S-", index, "S+", index + 2
        # @show index, index+2, J2
    end


    # Set up the Hamiltonian as MPOs and the initial wave function as MPS
    Hamiltonian = MPO(os, s)
    ψ₀ = MPS(s, n -> isodd(n) ? "Up" : "Dn")
    # states = [isodd(n) ? "Up" : "Dn" for n in 1:N]        # Neel state
    # ψ₀ = randomMPS(s, states; linkdims = 8)       # random MPS 
    
    
    # Tune the parameters used in DMRG to obtain the ground-state wave function
    nsweeps = 2
    eigsolve_krylovdim = 50
    maxdim = [20, 50, 200, 2000]
    E, ψ = dmrg(Hamiltonian, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim)
    
    
    # Measure physically relevant observables from the ground-state wave function
    # One-point, two-point functions
    Sz₀ = expect(ψ, "Sz"; sites=1:N)
    Czz₀ = correlation_matrix(ψ, "Sz", "Sz"; sites=1:N)
   
    # von Neummann entanglement entropy
    SvN = entanglement_entropy(ψ, N)
    # @show SvN

    # Bond dimensions
    chi = linkdims(ψ)
    # @show chi
    #*************************************************************************************************************************
    #*************************************************************************************************************************

    
    # Make gates (1, 2), (2, 3), ..., (N-1, N)
    gates = ITensor[]
    gates_bonds_with_disorders = 0
    for index in 1 : N - 2
        s₁ = s[index]
        s₂ = s[index + 1]
        s₃ = s[index + 2]

        random_number = random_numbers[index]
        if random_number < 0.1
            normalizedJ = J1 * 0.6
            gates_bonds_with_disorders += 1
            @show index, normalizedJ, random_number
        else
            normalizedJ = J1
        end
        @show index, normalizedJ

        if isodd(index)
            effectiveJ = normalizedJ * (1 + delta)
        else
            effectiveJ = normalizedJ * (1 - delta)
        end

        # Add two-site gate for nearest-neighbor interactions with bond disorders
        hj = 1/2 * effectiveJ * op("S+", s₁) * op("S-", s₂) + 1/2 * effectiveJ * op("S-", s₁) * op("S+", s₂) + effectiveJ * op("Sz", s₁) * op("Sz", s₂)
        Gj = exp(-im * τ/2 * hj)
        push!(gates, Gj)


        # Add two-site gate for next-nearest-neighbor interactions
        hj_tmp = 1/2 * J2 * op("S+", s₁) * op("S-", s₃) + 1/2 * J2 * op("S-", s₁) * op("S+", s₃) + J2 * op("Sz", s₁) * op("Sz", s₃) 
        Gj_tmp = exp(-im * τ/2 * hj_tmp)    
        push!(gates, Gj_tmp)
    end

    # Add the last gate for the last two sites
    s₁ = s[N - 1]
    s₂ = s[N]
    random_number = random_numbers[N - 1]
    if random_number < 0.1
        normalizedJ = J1 * 0.6
        gates_bonds_with_disorders += 1
        @show normalizedJ, random_number
    else
        normalizedJ = J1
    end
    # @show N-1, random_numbers[N - 1], normalizedJ
    if isodd(N - 1)
        effectiveJ = normalizedJ * (1 + delta)
    else
        effectiveJ = normalizedJ * (1 - delta)
    end
    hj = 1/2 * effectiveJ * op("S+", s₁) * op("S-", s₂) + 1/2 * effectiveJ * op("S-", s₁) * op("S+", s₂) + effectiveJ * op("Sz", s₁) * op("Sz", s₂)
    Gj = exp(-im * τ/2 * hj) 
    push!(gates, Gj)
    
    # Add reverse gates due to the the symmetric Trotter decomposition
    append!(gates, reverse(gates))

    if bonds_with_disorders != gates_bonds_with_disorders
        error("The number of bonds with disorders in gates is not consistent with that in the Hamiltonian!")
    end

    #*************************************************************************************************************************
    #************************************************************************************************************************* 
    # Apply a local perturbation at the center of two chains that are needed due to the dimmerization
    center₁, center₂ = div(N, 2), div(N, 2) - 1
    ψ_odd  = deepcopy(ψ)
    ψ_even = deepcopy(ψ)
    
    # Apply a local operator Sz to the even site in the center of the chain
    local_operator = op("Sz", s[center₁])
    ψ_even = apply(local_operator, ψ_even; cutoff)

    # Apply a local operator Sz to the odd site in the center of the chain    
    local_operator = op("Sz", s[center₂])
    ψ_odd = apply(local_operator, ψ_odd; cutoff)

    
    # Initialize the arrays to store the physical observables at different time steps
    Czz = zeros(ComplexF64, time_steps + 1, N * N)
    Czz_odd = zeros(ComplexF64, time_steps + 1, N * N)
    Czz_even = zeros(ComplexF64, time_steps + 1, N * N)
    Sz = zeros(ComplexF64, time_steps + 1, N)
    Sz_odd = zeros(ComplexF64, time_steps + 1, N)
    Sz_even = zeros(ComplexF64, time_steps + 1, N)
    Czz_unequaltime_odd  = zeros(ComplexF64, time_steps, N) 
    Czz_unequaltime_even = zeros(ComplexF64, time_steps, N)
    chi = zeros(Float64, time_steps + 1, N - 1)
    
    
    # Calculate the physical observables at different time steps
    Sz[1, :] = expect(ψ, "Sz"; sites = 1 : N)
    Sz_odd[1, :] = expect(ψ_odd, "Sz"; sites = 1 : N)
    Sz_even[1, :] = expect(ψ_even, "Sz"; sites = 1 : N)
    Czz[1, :] = correlation_matrix(ψ, "Sz", "Sz"; sites = 1 : N)
    Czz_odd[1, :] = correlation_matrix(ψ_odd, "Sz", "Sz"; sites = 1 : N)
    Czz_even[1, :] = correlation_matrix(ψ_even, "Sz", "Sz"; sites = 1 : N)
    chi[1, :] = linkdims(ψ)
    # @show chi[1, :]
    #*************************************************************************************************************************
    #************************************************************************************************************************* 
    

    #*************************************************************************************************************************
    #************************************************************************************************************************* 
    # Time evovle wavefunctions using TEBD
    for t in 0 : τ : ttotal
        index = round(Int, t / τ) + 1
        Sz₁, Sz₂, Sz₃ = expect(ψ, "Sz"; sites = center₂ : center₁), 
            expect(ψ_odd, "Sz"; sites = center₂ : center₁), expect(ψ_even, "Sz"; sites = center₂ : center₁)
        @show t, index, Sz₁, Sz₂, Sz₃
        t ≈ ttotal && break
        
        # Time evolve the pertubed and original wave functions
        ψ_odd = apply(gates, ψ_odd; cutoff) 
        normalize!(ψ_odd)
        
        ψ_even = apply(gates, ψ_even; cutoff)
        normalize!(ψ_even)
       
        ψ = apply(gates, ψ; cutoff)
        normalize!(ψ)
        chi[index + 1, :] = linkdims(ψ)
        @show index, t + τ

        # Compute the physically relevant observables at different time steps
        Czz[index + 1, :] = correlation_matrix(ψ, "Sz", "Sz"; sites = 1 : N)
        Czz_odd[index + 1, :] = correlation_matrix(ψ_odd, "Sz", "Sz"; sites = 1 : N)    
        Czz_even[index + 1, :] = correlation_matrix(ψ_even, "Sz", "Sz"; sites = 1 : N)  
        Sz[index + 1, :] = expect(ψ, "Sz"; sites = 1 : N)
        Sz_odd[index + 1, :] = expect(ψ_odd, "Sz"; sites = 1 : N)
        Sz_even[index + 1, :] = expect(ψ_even, "Sz"; sites = 1 : N)


        # Compute the unequal-time spin correlation function <Sz_i(t) Sz_j(0)>
        for site_index in collect(1 : N)
            tmp_os = OpSum()
            tmp_os += "Sz", site_index
            tmp_MPO = MPO(tmp_os, s)
            Czz_unequaltime_even[index, site_index] = inner(ψ', tmp_MPO, ψ_even)
            Czz_unequaltime_odd[index, site_index] = inner(ψ', tmp_MPO, ψ_odd)
        end

        # Create a HDF5 file and save the unequal-time spin correlation to the file at every time step
        h5open("data/heisenberg_disorder_N$(N)_J2$(J2)_delta$(delta).h5", "w") do file
            if haskey(file, "Czz_unequaltime_odd")
                delete_object(file, "Czz_unequaltime_odd")
            end
            write(file, "Czz_unequaltime_odd",  Czz_unequaltime_odd)

            if haskey(file, "Czz_unequaltime_even")
                delete_object(file, "Czz_unequaltime_even")
            end
            write(file, "Czz_unequaltime_even", Czz_unequaltime_even)
        end
    end
    #*************************************************************************************************************************
    #*************************************************************************************************************************


    h5open("data/heisenberg_binomial_disorder_N$(N)_v$(random_seed).h5", "w") do file
        write(file, "Psi", ψ)
        write(file, "Energy", E)
        write(file, "Bond", chi)
        write(file, "Sz", Sz)
        write(file, "Sz odd", Sz_odd)
        write(file, "Sz even", Sz_even)
        write(file, "Czz", Czz)
        write(file, "Czz odd", Czz_odd)
        write(file, "Czz even", Czz_even)
    end

    return
end