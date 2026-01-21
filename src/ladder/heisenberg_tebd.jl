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
    s = siteinds("S=1/2", N; conserve_qns=true)


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
            # @show i, x₁, y₁, j, x₂, y₂, dimerization_sign, J_effective
        end
        

        # Set up next-neareest-neighbor interactions along the horizontal direction
        if abs(i - j) == 4
            os .+= 0.5 * J2, "S+", i, "S-", j 
            os .+= 0.5 * J2, "S-", i, "S+", j 
            os .+= J2, "Sz", i, "Sz", j 
        end


        # Set up the interactions along the vertical direction
        if abs(i - j) == 1
            os .+= 0.5 * Jp, "S+", i, "S-", j 
            os .+= 0.5 * Jp, "S-", i, "S+", j
            os .+= Jp, "Sz", i, "Sz", j
        end
    end


    # Construct the Hamiltonian MPO and initialize the wave function as a random MPS
    Hamiltonian = MPO(os, s)
    

    # Initialize the wave function as an MPS
    states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    ψ₀ = randomMPS(s, states; linkdims = 8)     # Initialize a random MPS
    # ψ₀ = MPS(s, n -> isodd(n) ? "Up" : "Dn")  # Initialize a prodcut state 
    


    # Define parameters that are used in the DMRG optimization process
    println("\n" * repeat("=", 200))
    println("Running DMRG algorithms to obtain the ground state of the J₁-J₂-δ Heisenberg ladder model...")
    println(repeat("=", 200))
    nsweeps = 10
    maxdim = [20, 50, 200, 1000]
    eigsolve_krylovdim = 100
    E, ψ = dmrg(Hamiltonian, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim)
    Sz₀ = expect(ψ, "Sz"; sites=1:N)
    Czz₀ = correlation_matrix(ψ, "Sz", "Sz"; sites=1:N)
    println(repeat("=", 200))
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

    

    
    # # 08/14/2024
    # # Apply two perturbations to restore the translational invariance of the system in the existence of the dimmerized interactions
    # # Assuming the number of sites is even
    # center = div(N, 2)
    # center_odd = center - 1
    # @show center, center_odd
    # ψ_odd = deepcopy(ψ)
    # ψ_copy = deepcopy(ψ)


    # # Apply a local operator Sz to the two sites at the center of the chain
    # local_op = op("Sz", s[center])
    # ψ = apply(local_op, ψ; cutoff)  
    # # normalize!(ψ)

    # # Calculate the physical observables at different time steps
    # # @t=0
    # Sz₁ = expect(ψ, "Sz"; sites = 1 : N)
    # Czz₁ = correlation_matrix(ψ, "Sz", "Sz"; sites = 1 : N)
    # @show Sz₁
    # @show Czz₁



    # local_operator_odd = op("Sz", s[center_odd])
    # ψ_odd = apply(local_operator_odd, ψ_odd; cutoff)
    # # normalize!(ψ_odd)

    # # local_op = op("Sz", s[center])
    # # @show typeof(local_op)
    # # newA = local_op * ψ[center]
    # # newA = noprime(newA)
    # # ψ[center] = newA

    # # os_local = OpSum()
    # # os_local += 1/2, "Sz", center
    # # local_op = MPO(os_local, s)
    # # ψ = apply(local_op, ψ; cutoff)
    # # normalize!(ψ)
   
    
    # # 08/14/2024
    # # Calculate the physical observables at different time steps
    # # @t>0
    # Czz = Matrix{ComplexF64}(undef, Int(ttotal / τ), N * N)
    # Czz_odd = Matrix{ComplexF64}(undef, Int(ttotal / τ), N * N)
    # Czz_even = Matrix{ComplexF64}(undef, Int(ttotal / τ), N * N)
    # Czz_unequaltime_odd  = Matrix{ComplexF64}(undef, Int(ttotal / τ), N) 
    # Czz_unequaltime_even = Matrix{ComplexF64}(undef, Int(ttotal / τ), N)
    # chi = Matrix{Float64}(undef, Int(ttotal / τ), N - 1)
    # Sz_all = Matrix{ComplexF64}(undef, Int(ttotal / τ), N)
    # Sz_all_odd = Matrix{ComplexF64}(undef, Int(ttotal / τ), N)
    # Sz_all_even = Matrix{ComplexF64}(undef, Int(ttotal / τ), N)
    # @show size(Czz_unequaltime_odd), size(Czz_unequaltime_even), size(chi)
    

    # # Time evovle the original and perturbed wave functions
    # for t in 0 : τ : ttotal
    #     index = round(Int, t / τ) + 1
    #     @show index
    #     Sz = expect(ψ, "Sz"; sites = center)
    #     println("t = $t, Sz = $Sz")

    #     t ≈ ttotal && break
    #     # Time evolve the perturbed wave function with the perturbation applied on the even site in the center of the chain
    #     ψ = apply(gates, ψ; cutoff)
    #     normalize!(ψ)
    #     chi[index, :] = linkdims(ψ)
    #     @show linkdims(ψ)

    #     # Time evolve the perturbed wave function with the perturbation applied on the odd site in the center of the chain
    #     ψ_odd = apply(gates, ψ_odd; cutoff)
    #     normalize!(ψ_odd)

    #     # Time evolve the original wave function
    #     ψ_copy = apply(gates, ψ_copy; cutoff)
    #     normalize!(ψ_copy)

    #     Czz[index, :] = correlation_matrix(ψ, "Sz", "Sz"; sites = 1 : N)
    #     Czz_odd[index, :] = correlation_matrix(ψ_odd, "Sz", "Sz"; sites = 1 : N)    
    #     Czz_even[index, :] = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1 : N)  
    #     Sz_all[index, :] = expect(ψ_copy, "Sz"; sites = 1 : N)
    #     Sz_all_odd[index, :] = expect(ψ_odd, "Sz"; sites = 1 : N); @show expect(ψ_odd, "Sz"; sites = 1 : N)
    #     Sz_all_even[index, :] = expect(ψ, "Sz"; sites = 1 : N)


    #     # Calculate the unequaltime correlation function
    #     for site_index in collect(1 : N)
    #         tmp_os = OpSum()
    #         tmp_os += "Sz", site_index
    #         tmp_MPO = MPO(tmp_os, s)
    #         Czz_unequaltime_even[index, site_index] = inner(ψ_copy', tmp_MPO, ψ)
    #         Czz_unequaltime_odd[index, site_index] = inner(ψ_copy', tmp_MPO, ψ_odd)
    #     end

    #     # # Create a HDF5 file and save the unequal-time spin correlation to the file at every time step
    #     # h5open("/pscratch/sd/x/xiaobo23/TensorNetworks/spectral_function/J1_J2_Dimmerization/Delta0/data/Heisenberg_Dimerized_TEBD_Time$(ttotal)_J2_$(J2).h5", "w") do file
    #     #     if haskey(file, "Czz_unequaltime_odd")
    #     #         delete_object(file, "Czz_unequaltime_odd")
    #     #     end
    #     #     write(file, "Czz_unequaltime_odd",  Czz_unequaltime_odd)

    #     #     if haskey(file, "Czz_unequaltime_even")
    #     #         delete_object(file, "Czz_unequaltime_even")
    #     #     end
    #     #     write(file, "Czz_unequaltime_even", Czz_unequaltime_even)
    #     # end
    # end

    # h5open("/pscratch/sd/x/xiaobo23/TensorNetworks/spectral_function/J1_J2_Dimmerization/Delta0/data/Heisenberg_Dimerized_TEBD_Time$(ttotal)_J2_$(J2).h5", "r+") do file
    #     write(file, "Psi", ψ)
    #     write(file, "Sz T=0", Sz₀)
    #     write(file, "Czz T=0", Czz₀)
    #     write(file, "Sz Perturbed", Sz₁)
    #     write(file, "Czz Perturbed", Czz₁)
    #     write(file, "Czz", Czz)
    #     write(file, "Czz Odd", Czz_odd) 
    #     write(file, "Czz Even", Czz_even)
    #     write(file, "Sz", Sz_all)
    #     write(file, "Sz Odd", Sz_all_odd)
    #     write(file, "Sz Even", Sz_all_even)
    #     write(file, "Bond", chi)
    # end

    return
end