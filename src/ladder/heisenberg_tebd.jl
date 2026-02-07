# 1/20/2026
# Time-evolving block-decimation  (TEBD) for simulation of real-time dynamics of the J₁-J₂-δ Heisenberg ladder 

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


# Define simulation parameters used to set up the Hamiltonian and TEBD time evolution
const Nx = 50
const Ny = 2
const N = Nx * Ny - 2   # Total number of sites in the ladder lattice by removing the corner sites
const J1 = 1.0
const J2 = 0.35
const Jp=0.0
const delta = 0.04
const τ = 0.1
const ttotal = 100.0
const cutoff = 1e-10


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
    """
        Running DMRG simulation to obtain the ground-state wave function
    """
    
    # Generate the ladder lattice
    lattice = ladder_lattice(Nx, Ny; yperiodic=false)    
    # for (idx, bond) in enumerate(lattice)
    #     @info "Lattice Bond" site1=bond.s1 site2=bond.s2
    # end

    # Make an array of site indices
    s = siteinds("S=1/2", N; conserve_qns=false)

    # Construct the Hamiltonian using OpSum
    os = OpSum()
    for bond in lattice 
        i, j = bond.s1, bond.s2

        # Set up nearest-neighbor interactions along the horizontal direction
        if i == 1 && abs(j - i) == 1
            os .+= 0.5 * J1 * (1 + delta), "S+", i, "S-", j 
            os .+= 0.5 * J1 * (1 + delta), "S-", i, "S+", j 
            os .+= J1 * (1 + delta), "Sz", i, "Sz", j 
            # @info "Bond" site_i=i site_j=j type="Nearest neighbor" dimerization=1 J=J1*(1 + delta)
        else
            if abs(i - j) == 2
                x₁, x₂ = div(i - 2, Ny) + 2, div(j - 2, Ny) + 2
                y₁, y₂ = mod(i - 2, Ny) + 1, mod(j - 2, Ny) + 1
                
                # Determine the effective interaction strength considering the dimerization effect
                dimerization_sign = (y₁ == 1) == isodd(x₁) ? 1 : -1
                J_effective = J1 * (1 + dimerization_sign * delta)

                # Add the interaction terms to OpSum
                os .+= 0.5 * J_effective, "S+", i, "S-", j 
                os .+= 0.5 * J_effective, "S-", i, "S+", j
                os .+= J_effective, "Sz", i, "Sz", j
                # @info "Bond" site_i=i site_j=j type="Nearest neighbor" dimerization=dimerization_sign J=J_effective
            end
        end

        # Set up next-neareest-neighbor interactions along the horizontal direction
        if (i == 1 && abs(j - i) == 3) || (abs(j - i) == 4)
            os .+= 0.5 * J2, "S+", i, "S-", j 
            os .+= 0.5 * J2, "S-", i, "S+", j 
            os .+= J2, "Sz", i, "Sz", j 
            # @info "Bond" site_i=i site_j=j type="Next-nearest neighbor" J=J2
        end

        # Set up inter-chain coupling along the vertrical direction
        if i != 1 && abs(j - i) == 1
            os .+= 0.5 * Jp, "S+", i, "S-", j 
            os .+= 0.5 * Jp, "S-", i, "S+", j
            os .+= Jp, "Sz", i, "Sz", j
            # @info "Bond" site_i=i site_j=j type="Vertical" J=Jp
        end
    end


    # Construct the Hamiltonian MPO
    Hamiltonian = MPO(os, s)
    
    # Initialize the wave function as an MPS
    states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    ψ₀ = randomMPS(s, states; linkdims = 8)     # Initialize a random MPS
    # ψ₀ = MPS(s, n -> isodd(n) ? "Up" : "Dn")  # Initialize a prodcut state 


    # Define parameters that are used in the DMRG optimization process
    println("\n" * repeat("=", 200))
    println("Running DMRG algorithms to obtain the ground state of the J₁-J₂-δ Heisenberg ladder model...")
    println(repeat("=", 200))
    nsweeps = 15
    maxdim = [20, 50, 200, 1000]
    eigsolve_krylovdim = 50
    E, ψ = dmrg(Hamiltonian, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim)
    Sz₀ = expect(ψ, "Sz"; sites=1:N)
    Czz₀ = correlation_matrix(ψ, "Sz", "Sz"; sites=1:N)
    @show Sz₀
    @show linkdims(ψ)
    println(repeat("=", 200))


    # Save the final wave functions and observables to an HDF5 file 
    output_file = "../data/heisenberg_J2$(J2)_delta$(delta)_Jp$(Jp).h5" 
    
    h5open(output_file, "cw") do file
        write(file, "energy", E)
        write(file, "Sz", Sz₀)
        write(file, "Czz", Czz₀)
    end
    # #**************************************************************************************************************
    # #************************************************************************************************************** 



   
    #**************************************************************************************************************
    #************************************************************************************************************** 
    """
        Setting up two-qubit gates for TEBD time evolution
    """

    gates = ITensor[]
    
    # Set up two-qubit gates for nearest-neighbot interactions along the vertical direction 
    for index in 2 : 2 : N - 2
        # Check the largest index to avoid out-of-bound error
        if index + 1 > N 
            break
        end

        s₁ = s[index]
        s₂ = s[index + 1]

        hj = 1/2 * Jp * op("S+", s₁) * op("S-", s₂) + 1/2 * Jp * op("S-", s₁) * op("S+", s₂) + Jp * op("Sz", s₁) * op("Sz", s₂)
        Gj = exp(-im * τ/2 * hj)
        push!(gates, Gj)
        @info "Two-qubit gate for vertical bond added" site1=index site2=index+1
    end


    # Set up two-qubit gates for nearest-neighbor interactions along the horizontal direction
    # Set up the first gate containing the first site separately
    s₁ = s[1]
    s₂ = s[2]

    hj = 0.5 * J1 * (1 + delta) * op("S+", s₁) * op("S-", s₂) + 0.5 * J1 * (1 + delta) * op("S-", s₁) * op("S+", s₂) + J1 * (1 + delta) * op("Sz", s₁) * op("Sz", s₂)
    Gj = exp(-im * τ/2 * hj)
    push!(gates, Gj)
    # @info "Two-qubit gate for nearest-neighbor bond added" site1=1 site2=2 J1=J1*(1+delta)

    for offset in [3, 2, 4]
        for index in offset : 3 : N - 2
            # Check the largest index to avoid out-of-bound error
            if index + 2 > N 
                break
            end

            s₁ = s[index]
            s₂ = s[index + 2]
            
            x₁ = div(index - 2, Ny) + 2
            y₁ = mod(index - 2, Ny) + 1
            
            dimerization_sign = (y₁ == 1) == isodd(x₁) ? 1 : -1
            J_effective = J1 * (1 + dimerization_sign * delta)

            hj = 0.5 * J_effective * op("S+", s₁) * op("S-", s₂) + 0.5 * J_effective * op("S-", s₁) * op("S+", s₂) + J_effective * op("Sz", s₁) * op("Sz", s₂)
            Gj = exp(-im * τ/2 * hj)
            push!(gates, Gj)
            # @info "Two-qubit gate for nearest-neighbor bond added" site1=index site2=index+2 J=J_effective
        end
    end


    # Set up two-qubit gayes for next-nearest-neighbor interactions
    # Set up the first gate containing the first site separately
    s₁ = s[1]
    s₂ = s[4]
    hj = 0.5 * J2 * op("S+", s₁) * op("S-", s₂) + 0.5 * J2 * op("S-", s₁) * op("S+", s₂) + J2 * op("Sz", s₁) * op("Sz", s₂)
    Gj = exp(-im * τ/2 * hj)
    push!(gates, Gj)
    # @info "Two-qubit gate for next-nearest-neighbor bond added" site1=1 site2=4 J=J2

    starting_points = [5, 2, 3, 4, 6]

    for offset in starting_points
        for index in offset : 5 : N - 4
            # Check the largest index to avoid out-of-bound error
            if index + 4 > N 
                break 
            end

            s₁ = s[index]
            s₂ = s[index + 4]

            hj = 0.5 * J2 * op("S+", s₁) * op("S-", s₂) + 0.5 * J2 * op("S-", s₁) * op("S+", s₂) + J2 * op("Sz", s₁) * op("Sz", s₂)
            Gj = exp(-im * τ/2 * hj)
            push!(gates, Gj)
            # @info "Two-qubit gate for next-nearest-neighbor bond added" site1=index site2=index+4 J=J2
        end
    end


    # Add reverse gates due to the the symmetric Trotter decomposition
    append!(gates, reverse(gates))
    #**************************************************************************************************************
    #************************************************************************************************************** 

    
    #**************************************************************************************************************
    #************************************************************************************************************** 
    """
        Time evolution of the original and perturbed wave functions using TEBD algorithm
        Compute time-dependent observables and correlation functions
    """

    """
        Pre-allocate arrays for time-dependent observables
    """
    time_steps = Int(ttotal / τ)
    Sz  = [Matrix{ComplexF64}(undef, time_steps, N) for _ in 1:5]
    Czz = [Matrix{ComplexF64}(undef, time_steps, N) for _ in 1:5]
    chi = [Matrix{Int}(undef, time_steps, N - 1) for _ in 1:5]
    
    # Aliases for backward compatibility
    Sz₀, Sz₁, Sz₂, Sz₃, Sz₄ = Sz
    Czz₀, Czz₁, Czz₂, Czz₃, Czz₄ = Czz
    chi₀, chi₁, chi₂, chi₃, chi₄ = chi

    
    # Applying a local perturbation to copies of ground-state wave function and time evolve 
    # the perturbed wave functions
    # ψ = randomMPS(s, states; linkdims = 50)       # Initialize a random MPS
    # ψ = MPS(s, n -> isodd(n) ? "Up" : "Dn") 

    # Define four reference sites at the center of the ladder lattice
    center = div(N, 2)
    references = iseven(center) ? (center:center+3) : (center-1:center+2)
    println("\nReference sites for local perturbations")
    @show references=references

    
    # Create perturbed copies of the ground state
    perturbed_psi = [deepcopy(ψ) for _ in 1:length(references)]
    
    
    # Apply Sz perturbation to each copy at the corresponding reference site
    for (i, ref) in enumerate(references)
        perturbation = op("Sz", s[ref])
        perturbed_psi[i] = apply(perturbation, perturbed_psi[i]; cutoff)
        normalize!(perturbed_psi[i])
    end


    # Measure the original and perturbed wave functoins before time evolution && check the effects of perturbations
    Sz₀_initial = expect(ψ, "Sz"; sites = 1 : N)
    Sz₁_initial = expect(perturbed_psi[1], "Sz"; sites = 1 : N)
    Sz₂_initial = expect(perturbed_psi[2], "Sz"; sites = 1 : N)
    println("")
    @show Sz₀_initial[references[1] - 7 : references[1] + 7]
    println("")
    @show Sz₁_initial[references[1] - 7 : references[1] + 7]
    println("")
    @show Sz₂_initial[references[1] - 7 : references[1] + 7]


    Czz₀_initial = correlation_matrix(ψ, "Sz", "Sz"; sites = 1 : N)
    Czz₁_initial = correlation_matrix(perturbed_psi[1], "Sz", "Sz"; sites = 1 : N)
    # Czz₂_initial = correlation_matrix(perturbed_psi[2], "Sz", "Sz"; sites = 1 : N)
    # @show Czz₀_initial[references[1], references[1] - 7 : references[1] + 7]
    # @show Czz₁_initial[references[1], references[1] - 7 : references[1] + 7]
    # @show Czz₂_initial[references[1], references[1] - 7 : references[1] + 7]
   
    

    """
        Time evovle the original and perturbed wave functions using TEBD algorithm
        and compute time-dependent observables and correlation functions
    """
    for t in 0 : τ : ttotal
        index = Int(round(t / τ)) + 1
        t ≈ ttotal && break
        
        
        # Time evolve the unperturbed wave function
        ψ = apply(gates, ψ; cutoff=cutoff, maxdim=1000)
        normalize!(ψ)
             
        for i in 1 : length(references)
            perturbed_psi[i] = apply(gates, perturbed_psi[i]; cutoff=cutoff, maxdim=1000)
            normalize!(perturbed_psi[i])
        end
        @show "Time evolution" current_time=t+τ  
        @show linkdims(ψ)


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
    # **************************************************************************************************************
    # **************************************************************************************************************
    

    
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