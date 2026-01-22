# 1/20/2026
# Using time-evolving block decimation (TEBD) to simulate real-time dynamics of J₁-J₂-δ Heisenberg ladder 

using ITensors 
using ITensorMPS
using LinearAlgebra
using MKL
using HDF5

include("lattice.jl")



# Set up the number of threads for parallel computing 
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8  
OMP_NUM_THREADS = 8
@info "BLAS Configuration" BLAS.get_config()
@info "Number of BLAS threads" BLAS.get_num_threads()
@info "Number of Julia threads" Threads.nthreads()



# Define the model parameters as well as the time evolution parameters
const Nx = 10
const Ny = 2
const N = Nx * Ny
const J1 = 1.0
const J2 = 0.35
const Jp = 0.02
const delta = 0.04
const τ = 0.05
const ttotal = 0.1
const cutoff = 1E-10



let
    # Print simulation parameters
    println("\n" * repeat("=", 200))
    println("TEBD Simulation: J₁-J₂-δ Heisenberg Ladder Model")
    println(repeat("=", 200))
    println("Lattice:  Nx = $Nx, Ny = $Ny, N = $N")
    println("Couplings: J₁ = $J1, J₂ = $J2, Jₚ = $Jp, δ = $delta")
    println("Time evolution: τ = $τ, t_total = $ttotal, steps = $(Int(ttotal/τ))")
    println("Truncation cutoff: $cutoff")
    println(repeat("=", 200) * "\n")

    
    #**************************************************************************************************************
    #************************************************************************************************************** 
    # Running DMRG simulation to obtain the ground-state wave function
    
    
    # Generate the ladder lattice
    lattice = ladder_lattice(Nx, Ny; yperiodic=false)    


    # Make an array of site indices
    s = siteinds("S=1/2", N; conserve_qns=false)


    # Construct the Hamiltonian using OpSum
    os = OpSum()
    for bond in lattice 
        i, j = bond.s1, bond.s2
        

        # Set up nearest-neighbor interactions along the horizontal direction
        if abs(i - j) == 2
            x₁, x₂ = div(i - 1, Ny) + 1, div(j - 1, Ny) + 1
            y₁, y₂ = mod(i - 1, Ny) + 1, mod(j - 1, Ny) + 1
            
            # Determine the effective interaction strength considering the dimerization effect
            dimerization_sign = (y₁ == y₂ == 1) == isodd(x₁) ? 1 : -1
            J_effective = J1 * (1 + dimerization_sign * delta)

            # Add the interaction terms to OpSum
            os .+= 0.5 * J_effective, "S+", i, "S-", j 
            os .+= 0.5 * J_effective, "S-", i, "S+", j
            os .+= J_effective, "Sz", i, "Sz", j
            @info "Bond" site_i=i site_j=j type="Nearest neighbor" dimerization=dimerization_sign J=J_effective
        end
        

        # Set up next-neareest-neighbor interactions along the horizontal direction
        if abs(i - j) == 4
            os .+= 0.5 * J2, "S+", i, "S-", j 
            os .+= 0.5 * J2, "S-", i, "S+", j 
            os .+= J2, "Sz", i, "Sz", j 
            @info "Bond" site_i=i site_j=j type="Next-nearest neighbor" J=J2
        end


        # Set up the interactions along the vertical direction
        if abs(i - j) == 1
            os .+= 0.5 * Jp, "S+", i, "S-", j 
            os .+= 0.5 * Jp, "S-", i, "S+", j
            os .+= Jp, "Sz", i, "Sz", j
            @info "Bond" site_i=i site_j=j type="Vertical" J=Jp
        end
    end


    # Construct the Hamiltonian MPO and initialize the wave function as a random MPS
    Hamiltonian = MPO(os, s)
    

    # Initialize the wave function as an MPS
    states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    ψ₀ = randomMPS(s, states; linkdims = 8)     # Initialize a random MPS
    # ψ₀ = MPS(s, n -> isodd(n) ? "Up" : "Dn")  # Initialize a prodcut state 
    


    # # Define parameters that are used in the DMRG optimization process
    # println("\n" * repeat("=", 200))
    # println("Running DMRG algorithms to obtain the ground state of the J₁-J₂-δ Heisenberg ladder model...")
    # println(repeat("=", 200))
    # nsweeps = 2
    # maxdim = [20, 50, 200, 1000]
    # eigsolve_krylovdim = 100
    # E, ψ = dmrg(Hamiltonian, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim)
    # Sz₀ = expect(ψ, "Sz"; sites=1:N)
    # Czz₀ = correlation_matrix(ψ, "Sz", "Sz"; sites=1:N)
    # println(repeat("=", 200))
    #**************************************************************************************************************
    #************************************************************************************************************** 



   
    #**************************************************************************************************************
    #************************************************************************************************************** 
    # Construct the TEBD gates for time evolution
    gates = ITensor[]
    
    # Add two-qubit gate for interactions along the vertical direction 
    for index in 1 : 2 : N 
        s₁ = s[index]
        s₂ = s[index + 1]

        hj = 1/2 * Jp * op("S+", s₁) * op("S-", s₂) + 1/2 * Jp * op("S-", s₁) * op("S+", s₂) + Jp * op("Sz", s₁) * op("Sz", s₂)
        Gj = exp(-im * τ/2 * hj)
        push!(gates, Gj)
    end


    # Add two-qubit gates for nearest-neighbor interactions along the horizontal direction
    for offset in 1:3
        for index in offset:3:N-2
            s₁ = s[index]
            s₂ = s[index + 2]
            
            x₁ = div(index - 1, Ny) + 1
            y₁ = mod(index - 1, Ny) + 1
            
            dimerization_sign = (y₁ == 1) == isodd(x₁) ? 1 : -1
            J_effective = J1 * (1 + dimerization_sign * delta)

            hj = 0.5 * J_effective * op("S+", s₁) * op("S-", s₂) + 0.5 * J_effective * op("S-", s₁) * op("S+", s₂) + J_effective * op("Sz", s₁) * op("Sz", s₂)
            Gj = exp(-im * τ/2 * hj)
            push!(gates, Gj)
        end
    end


    # Add two-qubit gates for nearest-neighbor interactions along the horizontal direction
    for offset in 1:5
        for index in offset:5:N-5
            s₁ = s[index]
            s₂ = s[index + 4]

            hj = 0.5 * J2 * op("S+", s₁) * op("S-", s₂) + 0.5 * J2 * op("S-", s₁) * op("S+", s₂) + J2 * op("Sz", s₁) * op("Sz", s₂)
            Gj = exp(-im * τ/2 * hj)
            push!(gates, Gj)
        end
    end


    # Add reverse gates due to the the symmetric Trotter decomposition
    append!(gates, reverse(gates))
    #**************************************************************************************************************
    #************************************************************************************************************** 

    
    #**************************************************************************************************************
    #************************************************************************************************************** 
    # Applying a local perturbation to copies of ground-state wave function and time evolve 
    # the perturbed wave functions
    # ψ = randomMPS(s, states; linkdims = 50)       # Initialize a random MPS
    ψ = MPS(s, n -> isodd(n) ? "Up" : "Dn") 

    # Define reference sites in the unit cell near the center of the ladder
    center_x = div(Nx, 2)
    references = [
        2 * (center_x - 2) + 1,  # reference₁
        2 * (center_x - 1),      # reference₂
        2 * (center_x - 1) + 1,  # reference₃
        2 * center_x             # reference₄
    ]
    @show references
    
    # Create perturbed copies of the ground state
    perturbed_psi = [deepcopy(ψ) for _ in 1:length(references)]
    
    
    # Apply Sz perturbation to each copy at the corresponding reference site
    for (i, ref) in enumerate(references)
        perturbation = op("Sx", s[ref])
        perturbed_psi[i] = apply(perturbation, perturbed_psi[i]; cutoff)
        normalize!(perturbed_psi[i])
    end


    # Calculate the physical observables at different time steps
    Sz₀ = expect(ψ, "Sz"; sites = 1 : N)
    Sz₁ = expect(perturbed_psi[1], "Sz"; sites = 1 : N)
    Sz₂ = expect(perturbed_psi[2], "Sz"; sites = 1 : N)
    @show Sz₀[6 : 12]
    @show Sz₁[6 : 12]
    @show Sz₂[6 : 12]


    # Czz₀ = correlation_matrix(ψ, "Sz", "Sz"; sites = 1 : N)
    # Czz₁ = correlation_matrix(perturbed_psi[1], "Sz", "Sz"; sites = 1 : N)
    # Czz₂ = correlation_matrix(perturbed_psi[2], "Sz", "Sz"; sites = 1 : N)
    # @show Czz₀[6, 6 : 12]
    # @show Czz₁[6, 6 : 12]
    # @show Czz₂[6, 6 : 12]
    #**************************************************************************************************************
    #************************************************************************************************************** 
   


    #**************************************************************************************************************
    #**************************************************************************************************************
    # Time evolve the original and perturbed wave functions and record time-dependent observables


    # Initialize arrays to store time-dependent observables
    time_steps = Int(ttotal / τ)
    Sz₀ = Matrix{ComplexF64}(undef, time_steps, N)
    Sz₁ = Matrix{ComplexF64}(undef, time_steps, N)
    Sz₂ = Matrix{ComplexF64}(undef, time_steps, N)
    Sz₃ = Matrix{ComplexF64}(undef, time_steps, N)
    Sz₄ = Matrix{ComplexF64}(undef, time_steps, N)
    Czz₁ = Matrix{ComplexF64}(undef, time_steps, N)
    Czz₂ = Matrix{ComplexF64}(undef, time_steps, N)
    Czz₃ = Matrix{ComplexF64}(undef, time_steps, N)
    Czz₄ = Matrix{ComplexF64}(undef, time_steps, N)  
    chi₀ = Matrix{Int}(undef, time_steps, N - 1)  
    chi₁ = Matrix{Int}(undef, time_steps, N - 1)
    chi₂ = Matrix{Int}(undef, time_steps, N - 1)
    chi₃ = Matrix{Int}(undef, time_steps, N - 1)
    chi₄ = Matrix{Int}(undef, time_steps, N - 1)
    output_file = "heisenberg_tebd_time$(ttotal)_J2$(J2)_Jp$(Jp)_delta$(delta).h5"    



    # Time evovle the original and perturbed wave functions
    for t in 0 : τ : ttotal
        index = Int(round(t / τ)) + 1
        t ≈ ttotal && break
        
        
        # Time evolve the unperturbed wave function
        ψ = apply(gates, ψ; cutoff)
        normalize!(ψ)
             
        for i in 1 : length(references)
            perturbed_psi[i] = apply(gates, perturbed_psi[i]; cutoff)
            normalize!(perturbed_psi[i])
        end
        @info "Time evolution" current_time=t+τ step=index total_steps=time_steps max_bond_dim=maximum(linkdims(ψ))



        # Record time-dependent physical observables and bond dimensions
        Sz₀[index, :] = expect(ψ, "Sz"; sites = 1 : N)
        Sz₁[index, :] = expect(perturbed_psi[1], "Sz"; sites = 1 : N)
        Sz₂[index, :] = expect(perturbed_psi[2], "Sz"; sites = 1 : N)
        Sz₃[index, :] = expect(perturbed_psi[3], "Sz"; sites = 1 : N)
        Sz₄[index, :] = expect(perturbed_psi[4], "Sz"; sites = 1 : N)

        chi₀[index, :] = linkdims(ψ)
        chi₁[index, :] = linkdims(perturbed_psi[1])
        chi₂[index, :] = linkdims(perturbed_psi[2])
        chi₃[index, :] = linkdims(perturbed_psi[3])
        chi₄[index, :] = linkdims(perturbed_psi[4])


        # Compute time-dependent correlation functions
        for site_index in collect(1 : N)
            measurement_os = OpSum()
            measurement_os += "Sz", site_index
            measurement_mpo = MPO(measurement_os, s)

            Czz₁[index, site_index] = inner(ψ', measurement_mpo, perturbed_psi[1])
            Czz₂[index, site_index] = inner(ψ', measurement_mpo, perturbed_psi[2])
            Czz₃[index, site_index] = inner(ψ', measurement_mpo, perturbed_psi[3])
            Czz₄[index, site_index] = inner(ψ', measurement_mpo, perturbed_psi[4])
        end

        # Create/update HDF5 file using "cw" mode (create or read-write if exists)
        h5open(output_file, "cw") do file
            for (name, data) in [("Czz1", Czz₁), ("Czz2", Czz₂), ("Czz3", Czz₃), ("Czz4", Czz₄)]
                haskey(file, name) && delete_object(file, name)
                file[name] = data
            end
        end
    end

    #**************************************************************************************************************
    #**************************************************************************************************************
    

    
    # Save the final wave functions and observables to an HDF5 file
    h5open(output_file, "cw") do file
        write(file, "Psi0", ψ)
        write(file, "Psi1", perturbed_psi[1])
        write(file, "Psi2", perturbed_psi[2])
        write(file, "Psi3", perturbed_psi[3])
        write(file, "Psi4", perturbed_psi[4])
        write(file, "Sz0", Sz₀)
        write(file, "Sz1", Sz₁)
        write(file, "Sz2", Sz₂)
        write(file, "Sz3", Sz₃)
        write(file, "Sz4", Sz₄)
        write(file, "chi0", chi₀)
        write(file, "chi1", chi₁)
        write(file, "chi2", chi₂)
        write(file, "chi3", chi₃)
        write(file, "chi4", chi₄)
    end


    return
end