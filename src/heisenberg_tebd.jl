# 08/02/2024
# Use time evolving block decimation (TEBD) to simulate the real-time dynamics of the J₁-J₂ Heisenberg model    


using ITensors 
using ITensorMPS
using LinearAlgebra
using HDF5

MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8  
OMP_NUM_THREADS = 8


# Monitor the number of threads used by BLAS and LAPACK
@show BLAS.get_config()
@show BLAS.get_num_threads()


# Define the parameters for setting up the lattice and time evolution
const N = 10
const cutoff = 1e-12
const τ = 0.05
const ttotal = 10.0
const J1 = 1.0
const J2 = 0.0
# const delta = 0.0


let
    println(repeat("#", 200))
    println("Parameters used in the TEBD simulation...")
    @show N, cutoff, τ, ttotal, J1, J2
    println("")

    s = siteinds("S=1/2", N)
    # ψ = MPS(s, n -> isodd(n) ? "Up" : "Dn")
    
    
    # Initialize a random MPS as the initial state for TEBD time evolution
    Random.seed!(1234)
    ψ = random_mps(s, "↑"; linkdims=10)
    sz₀ = expect(ψ, "Sz"; sites=1:N)
    
    # # Read in the ground-state wave function of the Kitaev honeycomb model as the target MPS 
    # file = h5open("data/heisenberg_input_n10.h5", "r")
    # ψ = read(file, "Psi", MPS)
    # # @show typeof(ψ)
    # s = siteinds(ψ)
    # close(file)


    
    

    # Make gates (1, 2), (2, 3), ..., (N-1, N) for the Heisenberg model 
    gates = ITensor[]
    for index in 1 : N - 1
        s₁ = s[index]
        s₂ = s[index + 1]

        
        hj = 1/2 * J1 * op("S+", s₁) * op("S-", s₂) + 1/2 * J1 * op("S-", s₁) * op("S+", s₂) + J1 * op("Sz", s₁) * op("Sz", s₂)
        Gj = exp(-im * τ/2 * hj)
        push!(gates, Gj)
    end

    # Add reverse gates for the second-order Trotter decomposition
    append!(gates, reverse(gates))


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

    

    # Apply local perturbations to the ground-state wave function
    ψ_copy = deepcopy(ψ)
    center = div(N, 2)
    local_perturbation = op("Sz", s[center])
    ψ_copy = apply(local_perturbation, ψ_copy; cutoff)
    normalize!(ψ_copy)  


    # Calculate the physical observables at different time steps
    chi = Matrix{Float64}(undef, Int(ttotal / τ), N - 1)
    czz = Matrix{ComplexF64}(undef, Int(ttotal / τ), N * N)
    sz = Matrix{ComplexF64}(undef, Int(ttotal / τ), N)
    # Czz_unequaltime_odd  = Matrix{ComplexF64}(undef, Int(ttotal / τ), N) 
    # Czz_unequaltime_even = Matrix{ComplexF64}(undef, Int(ttotal / τ), N)
    

    # Time evovle the original and perturbed wave functions
    for t in 0 : τ : ttotal
        index = round(Int, t / τ) + 1
        @show index

        t ≈ ttotal && break


        # Time evolve the perturbed wave function with the perturbation applied on the odd site in the center of the chain
        ψ_copy = apply(gates, ψ_copy; cutoff)
        normalize!(ψ_copy)
        
        
        chi[index, :] = linkdims(ψ_copy)
        sz[index, :] = expect(ψ_copy, "Sz"; sites = 1 : N)
        czz[index, :] = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1 : N)
        

        # # Calculate the unequaltime correlation function
        # for site_index in collect(1 : N)
        #     tmp_os = OpSum()
        #     tmp_os += "Sz", site_index
        #     tmp_MPO = MPO(tmp_os, s)
        #     Czz_unequaltime_even[index, site_index] = inner(ψ_copy', tmp_MPO, ψ)
        #     Czz_unequaltime_odd[index, site_index] = inner(ψ_copy', tmp_MPO, ψ_odd)
        # end
    end

    @show sz₀   
    
    output_filename = "data/heisenberg_tebd_N$(N)_random.h5"
    h5open(output_filename, "w") do file
        write(file, "bond", chi)
        write(file, "czz", czz)
        write(file, "sz", sz)
    end

    return
end