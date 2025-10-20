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
const N = 100
const τ = 0.05
const ttotal = 0.2
const cutoff = 1E-10
const J₁ = 1.0       # Antiferromagnetic coupling
const J₂ = 0.35      # No next-nearest-neighbor interactions
const delta = 0.04   # No dimmerization
const time_steps = Int(ttotal / τ)
const α = 1e-6
const disorder_percentage=0.05


let 
    println(repeat("#", 200))
    println(repeat("#", 200))
    println("Time evolve the perturbed ground state of the J1-J2 Heisenberg model with disorders on sites.")
    println("The parameters used in the simulation are:")
    @show N, cutoff, τ, ttotal, J₁, J₂, delta
    # @show pinning  
   
    
    # Set up the bonds with disorders in a controlled way by using the same random seed
    random_seed=1
    Random.seed!(random_seed * 100000 + 1234567)

    site_disorders = zeros(Int, Int(disorder_percentage * N))
    idx=1
    while idx <= length(site_disorders)
        random_number = rand(1 : N)
        if random_number != 0 && !(random_number in site_disorders)
            site_disorders[idx] = random_number
            idx += 1
        end
    end
    
    println("")
    println("The sites with disorders are: ")
    @show site_disorders
    println(repeat("#", 200))
    println(repeat("#", 200))
    println("")
    
    
    #*************************************************************************************************************************
    #*************************************************************************************************************************
    # Make an array of "site" indices
    s = siteinds("S=1/2", N; conserve_qns=false)

    # Set up nearest-neighbor interactions with disorders
    println(repeat("#", 200))
    println(repeat("#", 200))
    println("Distribution of bond disorders in setting up the Hamiltonian:")
    os = OpSum()
    sites_with_disorders = 0
    for index in 1 : N - 1
        if index in site_disorders || (index + 1) in site_disorders
            normalized_J₁ = α * J₁
            sites_with_disorders += 1
            if index in site_disorders
                @show index, normalized_J₁
            elseif (index + 1) in site_disorders
                @show index + 1, normalized_J₁
            end
        else
            normalized_J₁ = J₁
        end

        if isodd(index)
            effective_J₁ = normalized_J₁ * (1 + delta)
        else
            effective_J₁ = normalized_J₁ * (1 - delta)
        end

        os += effective_J₁, "Sz", index, "Sz", index + 1
        os += 1/2 * effective_J₁, "S+", index, "S-", index + 1
        os += 1/2 * effective_J₁, "S-", index, "S+", index + 1
    end
    println("")
    println("")

    # Set up next-nearest-neighbor interactions
    for index in 1 : N - 2
        if index in site_disorders || (index + 2) in site_disorders
            effective_J₂ = α * J₂
            if index in site_disorders
                @show index, effective_J₂
            elseif (index + 2) in site_disorders
                @show index + 2, effective_J₂
            end
        else
            effective_J₂ = J₂
        end
        os += effective_J₂, "Sz", index, "Sz", index + 2
        os += 1/2 * effective_J₂, "S+", index, "S-", index + 2
        os += 1/2 * effective_J₂, "S-", index, "S+", index + 2
    end
    println(repeat("#", 200))
    println(repeat("#", 200))
    println("")
    println("")
    
    
    #*************************************************************************************************************************
    #*************************************************************************************************************************
    # Set up the Hamiltonian and initial wave function, and perform DMRG simulation to obtain the ground-state wave function
    println(repeat("#", 200))
    println(repeat("#", 200))
    println("Running DMRG simulation:")
    
    
    Hamiltonian = MPO(os, s)
    # ψ₀ = MPS(s, n -> isodd(n) ? "Up" : "Dn")
    states = [isodd(n) ? "Up" : "Dn" for n in 1:N]    # Neel state
    ψ₀ = randomMPS(s, states; linkdims = 8)           # random MPS 
    

    # Tune the parameters used in DMRG to obtain the ground-state wave function
    nsweeps = 2
    eigsolve_krylovdim = 50
    maxdim = [20, 50, 200, 2000]
    E, ψ = dmrg(Hamiltonian, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim)
    println(repeat("#", 200))
    println(repeat("#", 200))
    println("")
    println("")

    
    # Measure physically relevant observables from the ground-state wave function
    # One-point, two-point functions
    # Splus  = expect(ψ, "S+"; sites=1:N)
    # Sminus = expect(ψ, "S-"; sites=1:N)
    # Sx₀ = 0.5 * (Splus + Sminus)
    # Sy₀ = -0.5im * (Splus - Sminus)
    
    Sx₀ = expect(ψ, "Sx"; sites=1:N)
    Sy₀ = -im * expect(ψ, "iSy"; sites=1:N)
    Sz₀ = expect(ψ, "Sz"; sites=1:N)
    Cxx₀ = correlation_matrix(ψ, "Sx", "Sx"; sites=1:N)
    Cyy₀ = -correlation_matrix(ψ, "iSy", "iSy"; sites=1:N)
    Czz₀ = correlation_matrix(ψ, "Sz", "Sz"; sites=1:N)
   
    # von Neummann entanglement entropy
    SvN = entanglement_entropy(ψ, N)
    # @show SvN

    # Bond dimensions
    chi₀ = linkdims(ψ)
    # @show chi
    
    # Create an HDF5 file to save the ground-state wave function and physical observables
    output_filename = "heisenberg_disorder_N$(N)_version$(random_seed).h5"
    h5open(output_filename, "cw") do file
        write(file, "Psi0", ψ)
        write(file, "Energy", E)
        write(file, "SvN", SvN)
        write(file, "Bond t=0", chi₀)
        write(file, "Sx t=0", Sx₀)
        write(file, "Sy t=0", Sy₀)
        write(file, "Sz t=0", Sz₀)
        write(file, "Czz t=0", Czz₀)
    end
    #*************************************************************************************************************************
    #*************************************************************************************************************************

    
    #*************************************************************************************************************************
    #*************************************************************************************************************************
    # Set up the gates used in the TEBD simulation
    # Make gates (1, 2), (2, 3), ..., (N-1, N)
    gates = ITensor[]
    gates_sites_with_disorders = 0
    println(repeat("#", 200))
    println(repeat("#", 200))
    println("Distribution of bond disorders in setting up the gates:")

    for index in 1 : N - 2
        s₁ = s[index]
        s₂ = s[index + 1]
        s₃ = s[index + 2]

        if index in site_disorders || (index + 1) in site_disorders
            normalized_J₁ = α * J₁
            gates_sites_with_disorders += 1
            if index in site_disorders
                @show index, normalized_J₁
            elseif (index + 1) in site_disorders
                @show index + 1, normalized_J₁
            end
        else
            normalized_J₁ = J₁
        end
        
        if isodd(index)
            effective_J₁ = normalized_J₁ * (1 + delta)
        else
            effective_J₁ = normalized_J₁ * (1 - delta)
        end

        if index in site_disorders || (index + 2) in site_disorders
            effective_J₂ = α * J₂
            if index in site_disorders
                @show index, effective_J₂
            elseif (index + 2) in site_disorders
                @show index + 2, effective_J₂
            end
        else
            effective_J₂ = J₂
        end

        # Add two-site gate for nearest-neighbor interactions with bond disorders
        hj = 1/2 * effective_J₁ * op("S+", s₁) * op("S-", s₂) + 1/2 * effective_J₁ * op("S-", s₁) * op("S+", s₂) + effective_J₁ * op("Sz", s₁) * op("Sz", s₂)
        Gj = exp(-im * τ/2 * hj)
        push!(gates, Gj)

        # Add two-site gate for next-nearest-neighbor interactions
        hj_tmp = 1/2 * effective_J₂ * op("S+", s₁) * op("S-", s₃) + 1/2 * effective_J₂ * op("S-", s₁) * op("S+", s₃) + effective_J₂ * op("Sz", s₁) * op("Sz", s₃)
        Gj_tmp = exp(-im * τ/2 * hj_tmp)    
        push!(gates, Gj_tmp)
    end

    
    # Add the last gate for the last two sites
    s₁ = s[N - 1]
    s₂ = s[N]
    if N - 1 in site_disorders || N in site_disorders
        normalized_J₁ = α * J₁
        gates_sites_with_disorders += 1
        if N - 1 in site_disorders
            @show N - 1, normalized_J₁
        elseif N in site_disorders
            @show N, normalized_J₁
        end
    else
        normalized_J₁ = J₁
    end

    if isodd(N - 1)
        effective_J₁ = normalized_J₁ * (1 + delta)
    else
        effective_J₁ = normalized_J₁ * (1 - delta)
    end
    hj = 1/2 * effective_J₁ * op("S+", s₁) * op("S-", s₂) + 1/2 * effective_J₁ * op("S-", s₁) * op("S+", s₂) + effective_J₁ * op("Sz", s₁) * op("Sz", s₂)
    Gj = exp(-im * τ/2 * hj) 
    push!(gates, Gj)
    println(repeat("#", 200))
    println(repeat("#", 200))
    println("")
    println("")
    

    # Add reverse gates due to the the symmetric Trotter decomposition
    append!(gates, reverse(gates))

    if sites_with_disorders != gates_sites_with_disorders
        error("The number of sites with disorders in time evolution is not consistent with that in the Hamiltonian!")
    end
    #*************************************************************************************************************************
    #*************************************************************************************************************************


    #*************************************************************************************************************************
    #************************************************************************************************************************* 
    # Apply a local perturbation at each site of the chain because disorders break the translational invariance
    center₁, center₂ = div(N, 2) - 1, div(N, 2)
    ψ_odd, ψ_even = deepcopy(ψ), deepcopy(ψ)
    ψy_odd, ψy_even = deepcopy(ψ), deepcopy(ψ)
    ψx_odd, ψx_even = deepcopy(ψ), deepcopy(ψ)

    
    # Apply local operators Sx, Sy, Sz to the central unit cell of the chain
    ψ_odd  = apply(op("Sz", s[center₁]), ψ_odd; cutoff)
    ψ_even = apply(op("Sz", s[center₂]), ψ_even; cutoff)

    ψy_odd  = apply(op("Sy", s[center₁]), ψy_odd; cutoff)
    ψy_even = apply(op("Sy", s[center₂]), ψy_even; cutoff)
    
    ψx_odd  = apply(op("Sx", s[center₁]), ψx_odd; cutoff)
    ψx_even = apply(op("Sx", s[center₂]), ψx_even; cutoff)


    # Initialize the arrays to store the physical observables at different time steps
    Czz = zeros(ComplexF64, time_steps + 1, N * N)
    Czz_odd = zeros(ComplexF64, time_steps + 1, N * N)
    Czz_even = zeros(ComplexF64, time_steps + 1, N * N)
    Sz = zeros(ComplexF64, time_steps + 1, N)
    Sz_odd = zeros(ComplexF64, time_steps + 1, N)
    Sz_even = zeros(ComplexF64, time_steps + 1, N)
    Cxx_time_odd = zeros(ComplexF64, time_steps, N)
    Cxx_time_even = zeros(ComplexF64, time_steps, N)
    Cyy_time_odd = zeros(ComplexF64, time_steps, N)
    Cyy_time_even = zeros(ComplexF64, time_steps, N)
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
    #*************************************************************************************************************************
    #************************************************************************************************************************* 
    

    #*************************************************************************************************************************
    #************************************************************************************************************************* 
    # Time evolve wavefunctions using TEBD
    println(repeat("#", 200))
    println(repeat("#", 200))
    println("Time evolving the perturbed wave functions using TEBD:")
    for t in 0 : τ : ttotal
        index = round(Int, t / τ) + 1
        Sz₁, Sz₂, Sz₃ = expect(ψ, "Sz"; sites = center₁ : center₂), 
            expect(ψ_odd, "Sz"; sites = center₁ : center₂), expect(ψ_even, "Sz"; sites = center₁ : center₂)
        @show t, index, Sz₁, Sz₂, Sz₃
        t ≈ ttotal && break
        
        # Time evolve the pertubed and original wave functions
        ψ_odd = apply(gates, ψ_odd; cutoff) 
        normalize!(ψ_odd)
        
        ψ_even = apply(gates, ψ_even; cutoff)
        normalize!(ψ_even)
       
        ψy_odd = apply(gates, ψy_odd; cutoff)
        normalize!(ψy_odd)

        ψy_even = apply(gates, ψy_even; cutoff)
        normalize!(ψy_even)

        ψx_odd = apply(gates, ψx_odd; cutoff)
        normalize!(ψx_odd)

        ψx_even = apply(gates, ψx_even; cutoff)
        normalize!(ψx_even)
        
        ψ = apply(gates, ψ; cutoff)
        normalize!(ψ)
        
        
        # Record the bond dimensions at each time step to estimate the computational cost     
        chi[index + 1, :] = linkdims(ψ)
        @show index, t + τ, chi[index + 1, :]
        println("")

        # Compute the physically relevant observables at different time steps
        Czz[index + 1, :] = correlation_matrix(ψ, "Sz", "Sz"; sites = 1 : N)
        Czz_odd[index + 1, :] = correlation_matrix(ψ_odd, "Sz", "Sz"; sites = 1 : N)    
        Czz_even[index + 1, :] = correlation_matrix(ψ_even, "Sz", "Sz"; sites = 1 : N)  
        Sz[index + 1, :] = expect(ψ, "Sz"; sites = 1 : N)
        Sz_odd[index + 1, :] = expect(ψ_odd, "Sz"; sites = 1 : N)
        Sz_even[index + 1, :] = expect(ψ_even, "Sz"; sites = 1 : N)


        # Compute the unequal-time spin correlation function <Sz_i(t) Sz_j(0)>
        for site_index in collect(1 : N)
            # Compute the unequal-time spin correlation function Czz_unequaltime(site_index, t) = <Sz_i(t) Sz_j(0)>
            tmp_Sz = OpSum()
            tmp_Sz += "Sz", site_index
            MPO_Sz = MPO(tmp_Sz, s)
            Czz_unequaltime_even[index, site_index] = inner(ψ', MPO_Sz, ψ_even)
            Czz_unequaltime_odd[index, site_index] = inner(ψ', MPO_Sz, ψ_odd)

            # Compute the unequal-time spin correlation function Cxx_time(site_index, t) = <Sx_i(t) Sx_j(0)>
            tmp_Sx = OpSum()
            tmp_Sx += "Sx", site_index
            MPO_Sx = MPO(tmp_Sx, s)
            Cxx_time_even[index, site_index] = inner(ψ', MPO_Sx, ψx_even)
            Cxx_time_odd[index, site_index] = inner(ψ', MPO_Sx, ψx_odd)

            # Compute the unequal-time spin correlation function Cyy_time(site_index, t) = <Sy_i(t) Sy_j(0)>
            tmp_Sy = OpSum()
            tmp_Sy += "Sy", site_index
            MPO_Sy = MPO(tmp_Sy, s)
            Cyy_time_even[index, site_index] = inner(ψ', MPO_Sy, ψy_even)
            Cyy_time_odd[index, site_index] = inner(ψ', MPO_Sy, ψy_odd)
        end

        
        # Save unequal-time spin correlations at each time step 
        h5open(output_filename, "cw") do file
            if haskey(file, "Cxx_time_odd")
                delete_object(file, "Cxx_time_odd")
            end
            write(file, "Cxx_time_odd",  Cxx_time_odd)

            if haskey(file, "Cxx_time_even")
                delete_object(file, "Cxx_time_even")
            end
            write(file, "Cxx_time_even", Cxx_time_even)


            if haskey(file, "Cyy_time_odd")
                delete_object(file, "Cyy_time_odd")
            end
            write(file, "Cyy_time_odd",  Cyy_time_odd)

            if haskey(file, "Cyy_time_even")
                delete_object(file, "Cyy_time_even")
            end
            write(file, "Cyy_time_even", Cyy_time_even)


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
    println(repeat("#", 200))
    println(repeat("#", 200))

    # Save the final results into the HDF5 file
    h5open(output_filename, "cw") do file
        write(file, "Psi", ψ)
        write(file, "Psi odd", ψ_odd)
        write(file, "Psi even", ψ_even)
        write(file, "Psi_x odd", ψx_odd)
        write(file, "Psi_x even", ψx_even)
        write(file, "Psi_y odd", ψy_odd)
        write(file, "Psi_y even", ψy_even)
        write(file, "Bond", chi)
        write(file, "Sz", Sz)
        write(file, "Sz odd", Sz_odd)
        write(file, "Sz even", Sz_even)
        write(file, "Czz", Czz)
        write(file, "Czz odd", Czz_odd)
        write(file, "Czz even", Czz_even)
    end
    #*************************************************************************************************************************
    #*************************************************************************************************************************

    return
end