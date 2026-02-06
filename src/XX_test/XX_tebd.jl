# 2/6/2026
# Running TEBD for real-time evolution of a XX spin chain model

using ITensors 
using ITensorMPS
using LinearAlgebra
using HDF5
using Random


# Set up the number of threads for parallel computing 
MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8  
OMP_NUM_THREADS = 8
@info "BLAS Configuration" BLAS.get_config()
@info "Number of BLAS threads" BLAS.get_num_threads()
@info "Number of Julia threads" Threads.nthreads()


# Define the parameters for setting up the lattice and time evolution
const N = 50
const cutoff = 1e-10
const τ = 0.05
const ttotal = 1.0
const J₁ = 1.0


let
    println(repeat("#", 200))
    println("Parameters used in the TEBD simulation...")
    @show N, cutoff, τ, ttotal, J₁
    println("")


    """
        Running DMRG to obtain the ground-state wave function and energy
    """
    
    # Initialize a random MPS as the initial state for TEBD time evolution
    Random.seed!(1234567)
    s = siteinds("S=1/2", N)
    ψ₀ = random_mps(s, "↑"; linkdims=10)
    sz₀ = expect(ψ₀, "Sz"; sites=1:N)
    # @show sz₀

    
    # Set up the Hamiltonian as an MPO for the XX model
    os = OpSum()
    for j in 1 : N - 1
        os .+= 1/2 * J₁, "S+", j, "S-", j+1
        os .+= 1/2 * J₁, "S-", j, "S+", j+1
    end
    H = MPO(os, s)
    

    println("\nRunning DMRG to obtain the ground-state wave function and energy...")
    nsweeps = 10
    maxdim = [20, 50, 200, 1000]
    eigsolve_krylovdim = 50
    E, ψ = dmrg(H, ψ₀; nsweeps, maxdim, cutoff, eigsolve_krylovdim)
    Sz₀_initial = expect(ψ, "Sz"; sites=1:N)
    Czz₀_initial = correlation_matrix(ψ, "Sz", "Sz"; sites=1:N)
    println("") 


    """
        Setting up MPOs/two-qubit gates for the TEBD time evolution using the second-order Trotter decomposition
    """

    # Make gates (1, 2), (2, 3), ..., (N-1, N) for the XX model
    gates = ITensor[]
    for index in 1 : N - 1
        s₁ = s[index]
        s₂ = s[index + 1]

        hj = 0.5 * J₁ * op("S+", s₁) * op("S-", s₂) + 0.5 * J₁ * op("S-", s₁) * op("S+", s₂) 
        Gj = exp(-im * τ/2 * hj)
        push!(gates, Gj)
    end

    # Add reverse gates for the second-order Trotter decomposition
    append!(gates, reverse(gates))


    
    """
        Time evolve the original and perturbed wave functions, and calculate the physical observables at different time steps
    """ 

    
    # Initialize matrices to store the time evolution data 
    chi = Matrix{Float64}(undef, Int(ttotal / τ), N - 1)
    sz = Matrix{ComplexF64}(undef, Int(ttotal / τ), N)
    czz = Matrix{ComplexF64}(undef, Int(ttotal / τ), N * N)
    czz_time = Matrix{ComplexF64}(undef, Int(ttotal / τ), N)

    
    # Create a copy of the original wave function for time evolution, and apply a local perturbation in the center of the chain
    ψ₁ = deepcopy(ψ)
    center = div(N, 2)
    perturbation = op("Sz", s[center])
    ψ₁ = apply(perturbation, ψ₁; cutoff)
    normalize!(ψ₁)  


    println("\nStarting TEBD time evolution...")
    # Time evovle the original and perturbed wave functions along real-time axis 
    for t in 0 : τ : ttotal
        index = round(Int, t / τ) + 1
        t ≈ ttotal && break
        println("\nReal-time evolution", " t_next=", round(t+τ, digits=2), " index=", index)
        @show linkdims(ψ₁)

        
        # Evolve the origina wave function ψ using TEBD 
        ψ = apply(gates, ψ; cutoff)
        normalize!(ψ)
        
        
        # Evolve the perturbed wave function ψ₁ using TEBD
        ψ₁ = apply(gates, ψ₁; cutoff)
        normalize!(ψ₁)
        
        
        # Measure the physical observables for the perturbed wave function 
        chi[index, :] = linkdims(ψ₁)
        sz[index, :]  = expect(ψ₁, "Sz"; sites = 1 : N)
        czz[index, :] = correlation_matrix(ψ₁, "Sz", "Sz"; sites = 1 : N)
       

        # Compute the unequal-time correlation function ⟨Sᶻ(t)Sᶻ(0)⟩ 
        for site_index in collect(1 : N)
            tmp_os = OpSum()
            tmp_os += "Sz", site_index
            tmp_mpo = MPO(tmp_os, s)
            czz_time[index, site_index] = inner(ψ', tmp_mpo, ψ₁)
        end
    end
    

    # Save the time evolution data to an HDF5 file 
    output_filename = "data/heisenberg_tebd_N$(N)_random.h5"
    output_dir = dirname(output_filename)
    
    # Create directory if it doesn't exist
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    h5open(output_filename, "w") do file
        write(file, "chi", chi)
        write(file, "sz", sz)
        write(file, "czz", czz)
        write(file, "czz_time", czz_time)
    end
    println(repeat("#", 200))

    return
end