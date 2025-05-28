# 08/02/2024
# Simulate real-time evolution of the one-dimensional dimerized J1-J2 Heisenberg model using time-evolving block decimation (TEBD)
# using ITensors 
using ITensorMPS
using LinearAlgebra
using MKL
using HDF5

MKL_NUM_THREADS = 8
OPENBLAS_NUM_THREADS = 8
OMP_NUM_THREADS = 8


let
    # Display BLAS configuration and number of threads
    @info "BLAS configuration:" BLAS.get_config()
    @info "BLAS number of threads:" BLAS.get_num_threads()


    # Define the parameters for setting up the lattice and time evolution
    N = 200
    cutoff = 1E-10
    τ = 0.05
    ttotal = 0.1
    
    # Define the dimmerazation parameter 
    J₁ = 1.0
    J₂ = 0.5
    δ  = 0.5

    println("")
    println("The parameters used in this simulation are:")
    @show N, cutoff, τ, ttotal, J₁, J₂, δ   
    println("")

    # Define the site indices for the spin-1/2 system
    s = siteinds("S=1/2", N; conserve_qns=true)
    
    # Make gates (1, 2), (2, 3), ..., (N-1, N)
    gates = ITensor[]
    for index in 1 : N - 2
        s₁ = s[index]
        s₂ = s[index + 1]
        s₃ = s[index + 2]

        # Add two-site gate for nearest-neighbor interactions
        if index % 2 == 1
            hj = 1/2 * J₁ * (1 + δ) * op("S+", s₁) * op("S-", s₂) + 1/2 * J₁ * (1 + δ) * op("S-", s₁) * op("S+", s₂) + J₁ * (1 + δ) * op("Sz", s₁) * op("Sz", s₂)
            Gj = exp(-im * τ/2 * hj)
        else
            hj = 1/2 * J₁ * (1 - δ) * op("S+", s₁) * op("S-", s₂) + 1/2 * J₁ * (1 - δ) * op("S-", s₁) * op("S+", s₂) + J₁ * (1 - δ) * op("Sz", s₁) * op("Sz", s₂)
            Gj = exp(-im * τ/2 * hj)
        end
        push!(gates, Gj)


        # Add two-site gate for next-nearest-neighbor interactions
        hj_tmp = 1/2 * J₂ * op("S+", s₁) * op("S-", s₃) + 1/2 * J₂ * op("S-", s₁) * op("S+", s₃) + J₂ * op("Sz", s₁) * op("Sz", s₃) 
        Gj_tmp = exp(-im * τ/2 * hj_tmp)    
        push!(gates, Gj_tmp)
    end

    # Add the last gate for the last two sites
    s₁ = s[N - 1]
    s₂ = s[N]
    if (N - 1) % 2 == 1
        hj = 1/2 * J₁ * (1 + δ) * op("S+", s₁) * op("S-", s₂) + 1/2 * J₁ * (1 + δ) * op("S-", s₁) * op("S+", s₂) + J₁ * (1 + δ) * op("Sz", s₁) * op("Sz", s₂)
        Gj = exp(-im * τ/2 * hj)
    else
        hj = 1/2 * J₁ * (1 - δ) * op("S+", s₁) * op("S-", s₂) + 1/2 * J₁ * (1 - δ) * op("S-", s₁) * op("S+", s₂) + J₁ * (1 - δ) * op("Sz", s₁) * op("Sz", s₂)
        Gj = exp(-im * τ/2 * hj)
    end
    push!(gates, Gj)
    
    # Add reverse gates due to the the symmetric Trotter decomposition
    append!(gates, reverse(gates))

    
    #******************************************************************************************************************************************************************
    # Compute the ground state of the dimerized Heisenberg model using DMRG
    #******************************************************************************************************************************************************************
    # Construct the Hamiltonian for the dimerized Heisenberg model as MPOs
    os = OpSum()
    for idx = 1 : N - 2
        # Construct the nearest-neighbor interactions with dimerization
        if isodd(idx)
            coeff_dimer = J₁ * (1 + δ)
        else
            coeff_dimer = J₁ * (1 - δ)
        end
        os += 0.5 * coeff_dimer, "S+", idx, "S-", idx + 1
        os += 0.5 * coeff_dimer, "S-", idx, "S+", idx + 1
        os += coeff_dimer,       "Sz", idx, "Sz", idx + 1

        # Construct the next-nearest-neighbor interactions
        os += 0.5 * J₂, "S+", idx, "S-", idx + 2
        os += 0.5 * J₂, "S-", idx, "S+", idx + 2
        os += J₂, "Sz", idx, "Sz", idx + 2
    end

    # Construct the MPO for the last two sites
    coeff_last = J₁ * (isodd(N - 1) ? (1 + δ) : (1 - δ))
    os += 0.5 * coeff_last, "S+", N - 1, "S-", N
    os += 0.5 * coeff_last, "S-", N - 1, "S+", N
    os += coeff_last,       "Sz", N - 1, "Sz", N
    #******************************************************************************************************************************************************************
    #******************************************************************************************************************************************************************

    # Start from an entangled state e.g. a random MPS  
    # states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    # ψ = randomMPS(s, states; linkdims = 10)

    
    # Starting from a product state and construct the Hamiltonian as an MPO
    ψ₀ = MPS(s, n -> isodd(n) ? "Up" : "Dn")
    Hamiltonian = MPO(os, s)
    
    # Define hyperparameters for DMRG simulation
    nsweeps = 20
    maxdim = [20, 50, 200, 2000]
    E, ψ = dmrg(Hamiltonian, ψ₀; nsweeps, maxdim, cutoff)

    # Measure one-point and two-point correlation functions of the ground state wave function
    Sz₀ = expect(ψ, "Sz"; sites=1:N)
    @show Sz₀
    # Czz₀ = correlation_matrix(ψ, "Sz", "Sz"; sites=1:N)
    # @show Czz₀
    #******************************************************************************************************************************************************************
    #****************************************************************************************************************************************************************** 

    # 08/14/2024
    # Apply perturbations to the ground state wave function to excite the system
    # Perturbations are applied to two sites at the center of the chain because the dimerization breaks the translational invariance of the system
    if isodd(N)
        error("The number of sites N must be even for the dimerized Heisenberg model!")
    end

    center1 = div(N, 2)
    center2 = center1 - 1
    @show center1, center2
    ψ_odd  = deepcopy(ψ)
    ψ_copy = deepcopy(ψ)

    # Apply a local operator Sz to the two sites at the center of the chain
    local_op = op("Sz", s[center1])
    ψ = apply(local_op, ψ; cutoff)  
    # normalize!(ψ)

    # Compute one-point and two-point functions after the perturbation
    Sz₁ = expect(ψ, "Sz"; sites = 1 : N)
    Czz₁ = correlation_matrix(ψ, "Sz", "Sz"; sites = 1 : N)
    # @show Sz₁
    # @show Czz₁

    local_operator_odd = op("Sz", s[center2])
    ψ_odd = apply(local_operator_odd, ψ_odd; cutoff)
    # normalize!(ψ_odd)
    
    #*******************************************************************************************************************************************************************
    # Test the application of local operators (e.g., S+ or S-) as perturbations to the wave function
    #*******************************************************************************************************************************************************************
    # ψ_tmp  = deepcopy(ψ)
    # local_tmp = op("S+", s[center1])
    # ψ_tmp = apply(local_tmp, ψ_tmp; cutoff)
    # normalize!(ψ_tmp)
   
    # Sz₂ = expect(ψ_tmp, "Sz"; sites = 1 : N)
    # @show Sz₂[95 : 105]
    # Sx₂ = expect(ψ_tmp, "Sx"; sites = 1 : N)
    # @show Sx₂[95 : 105]
    #********************************************************************************************************************************************************************
    #********************************************************************************************************************************************************************

    
    # Initialize arrays to store the physical observables at each step
    nsteps = Int(round(ttotal / τ)) + 1
    Czz               = zeros(ComplexF64, nsteps, N * N)
    Czz_odd           = zeros(ComplexF64, nsteps, N * N)
    Czz_even          = zeros(ComplexF64, nsteps, N * N)
    Czz_unequaltime_odd  = zeros(ComplexF64, nsteps, N)
    Czz_unequaltime_even = zeros(ComplexF64, nsteps, N)
    chi               = zeros(Float64, nsteps, N - 1)
    Sz_all            = zeros(ComplexF64, nsteps, N)
    Sz_all_odd        = zeros(ComplexF64, nsteps, N)
    Sz_all_even       = zeros(ComplexF64, nsteps, N)

    @info "Initialized arrays" 
        size_Czz_unequaltime_odd = size(Czz_unequaltime_odd), 
        size_Czz_unequaltime_even = size(Czz_unequaltime_even), 
        size_chi = size(chi)
    

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
    #     # h5open("Data/TDVP/Heisenberg_Dimerized_TEBD_Time$(ttotal)_Delta$(δ)_J2$(J₂).h5", "w") do file
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

    # h5open("Data/TDVP/Heisenberg_Dimerized_TEBD_Time$(ttotal)_Delta$(δ)_J2$(J₂).h5", "r+") do file
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