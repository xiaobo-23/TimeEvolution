# 08/02/2024
# Use time evolving block decimation (TEBD) to simulate the time evolution of the 1D J1-J2 Heisenberg model.    

using ITensors 
using ITensorMPS
using LinearAlgebra
using MKL
using HDF5

MKL_NUM_THREADS = 12
OPENBLAS_NUM_THREADS = 12   
OMP_NUM_THREADS = 12


let 
    # Monitor the number of threads used by BLAS and LAPACK
    @show BLAS.get_config()
    @show BLAS.get_num_threads()

    # Define the parameters for setting up the lattice and time evolution
    N = 200
    cutoff = 1E-10
    τ = 0.05
    ttotal = 2.0
    
    # Define the dimmerazation parameter 
    J₁ = 1.0
    J₂ = 0.5
    δ  = 0.5

    println("")
    println("The parameters used in this simulation are:")
    @show N, cutoff, τ, ttotal, J₁, J₂, δ   
    println("")

    # Make an array of "site" indices
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

    
    # Run DMRG simulation to obtain the ground-state wave function
    os = OpSum()
    for index = 1 : N - 2
        # Construct the Hamiltonian for the Heisenberg model
        # Consider the nearest-neighbor dimmerized interactions
        if index % 2 == 1
            os += 1/2 * J₁ * (1 + δ), "S+", index, "S-", index + 1
            os += 1/2 * J₁ * (1 + δ), "S-", index, "S+", index + 1
            os += J₁ * (1 + δ), "Sz", index, "Sz", index + 1
        else
            os += 1/2 * J₁ * (1 - δ), "S+", index, "S-", index + 1
            os += 1/2 * J₁ * (1 - δ), "S-", index, "S+", index + 1
            os += J₁ * (1 - δ), "Sz", index, "Sz", index + 1
        end

        # Consider the next-nearest-neighbor interactions
        os += 1/2 * J₂, "S+", index, "S-", index + 2    
        os += 1/2 * J₂, "S-", index, "S+", index + 2
        os += J₂, "Sz", index, "Sz", index + 2  
    end

    # Construct the MPO for the last two sites
    if (N - 1) % 2 == 1
        os += 1/2 * J₁ * (1 + δ), "S+", N - 1, "S-", N  
        os += 1/2 * J₁ * (1 + δ), "S-", N - 1, "S+", N
        os += J₁ * (1 + δ), "Sz", N - 1, "Sz", N
    else
        os += 1/2 * J₁ * (1 - δ), "S+", N - 1, "S-", N  
        os += 1/2 * J₁ * (1 - δ), "S-", N - 1, "S+", N
        os += J₁ * (1 - δ), "Sz", N - 1, "Sz", N
    end

    
    Hamiltonian = MPO(os, s)
    ψ₀ = MPS(s, n -> isodd(n) ? "Up" : "Dn")
    

    # Define parameters that are used in the DMRG optimization process
    nsweeps = 20
    maxdim = [20, 50, 200, 2000]
    states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    # ψ = randomMPS(s, states; linkdims = 10)
    E, ψ = dmrg(Hamiltonian, ψ₀; nsweeps, maxdim, cutoff)
    
    Sz₀ = expect(ψ, "Sz"; sites=1:N)
    Czz₀ = correlation_matrix(ψ, "Sz", "Sz"; sites=1:N)
    # @show Sz₀
    # @show Czz₀


    # 08/14/2024
    # Apply two perturbations to restore the translational invariance of the system in the existence of the dimmerized interactions
    # Assuming the number of sites is even
    center = div(N, 2)
    center_odd = center - 1
    @show center, center_odd
    ψ_odd = deepcopy(ψ)
    ψ_copy = deepcopy(ψ)


    # Apply a local operator Sz to the two sites at the center of the chain
    local_op = op("Sz", s[center])
    ψ = apply(local_op, ψ; cutoff)  
    # normalize!(ψ)

    # Calculate the physical observables at different time steps
    # @t=0
    Sz₁ = expect(ψ, "Sz"; sites = 1 : N)
    Czz₁ = correlation_matrix(ψ, "Sz", "Sz"; sites = 1 : N)
    @show Sz₁
    @show Czz₁



    local_operator_odd = op("Sz", s[center_odd])
    ψ_odd = apply(local_operator_odd, ψ_odd; cutoff)
    # normalize!(ψ_odd)

    # local_op = op("Sz", s[center])
    # @show typeof(local_op)
    # newA = local_op * ψ[center]
    # newA = noprime(newA)
    # ψ[center] = newA

    # os_local = OpSum()
    # os_local += 1/2, "Sz", center
    # local_op = MPO(os_local, s)
    # ψ = apply(local_op, ψ; cutoff)
    # normalize!(ψ)
   
    
    # 08/14/2024
    # Calculate the physical observables at different time steps
    # @t>0
    Czz = Matrix{ComplexF64}(undef, Int(ttotal / τ), N * N)
    Czz_odd = Matrix{ComplexF64}(undef, Int(ttotal / τ), N * N)
    Czz_even = Matrix{ComplexF64}(undef, Int(ttotal / τ), N * N)
    Czz_unequaltime_odd  = Matrix{ComplexF64}(undef, Int(ttotal / τ), N) 
    Czz_unequaltime_even = Matrix{ComplexF64}(undef, Int(ttotal / τ), N)
    chi = Matrix{Float64}(undef, Int(ttotal / τ), N - 1)
    Sz_all = Matrix{ComplexF64}(undef, Int(ttotal / τ), N)
    Sz_all_odd = Matrix{ComplexF64}(undef, Int(ttotal / τ), N)
    Sz_all_even = Matrix{ComplexF64}(undef, Int(ttotal / τ), N)
    @show size(Czz_unequaltime_odd), size(Czz_unequaltime_even), size(chi)
    

    # Time evovle the original and perturbed wave functions
    for t in 0 : τ : ttotal
        index = round(Int, t / τ) + 1
        @show index
        Sz = expect(ψ, "Sz"; sites = center)
        println("t = $t, Sz = $Sz")

        t ≈ ttotal && break
        # Time evolve the perturbed wave function with the perturbation applied on the even site in the center of the chain
        ψ = apply(gates, ψ; cutoff)
        normalize!(ψ)
        chi[index, :] = linkdims(ψ)
        @show linkdims(ψ)

        # Time evolve the perturbed wave function with the perturbation applied on the odd site in the center of the chain
        ψ_odd = apply(gates, ψ_odd; cutoff)
        normalize!(ψ_odd)

        # Time evolve the original wave function
        ψ_copy = apply(gates, ψ_copy; cutoff)
        normalize!(ψ_copy)

        Czz[index, :] = correlation_matrix(ψ, "Sz", "Sz"; sites = 1 : N)
        Czz_odd[index, :] = correlation_matrix(ψ_odd, "Sz", "Sz"; sites = 1 : N)    
        Czz_even[index, :] = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1 : N)  
        Sz_all[index, :] = expect(ψ_copy, "Sz"; sites = 1 : N)
        Sz_all_odd[index, :] = expect(ψ_odd, "Sz"; sites = 1 : N); @show expect(ψ_odd, "Sz"; sites = 1 : N)
        Sz_all_even[index, :] = expect(ψ, "Sz"; sites = 1 : N)


        # Calculate the unequaltime correlation function
        for site_index in collect(1 : N)
            tmp_os = OpSum()
            tmp_os += "Sz", site_index
            tmp_MPO = MPO(tmp_os, s)
            Czz_unequaltime_even[index, site_index] = inner(ψ_copy', tmp_MPO, ψ)
            Czz_unequaltime_odd[index, site_index] = inner(ψ_copy', tmp_MPO, ψ_odd)
        end

        # Create a HDF5 file and save the unequal-time spin correlation to the file at every time step
        h5open("Data/TDVP/Heisenberg_Dimerized_TEBD_Time$(ttotal)_Delta$(δ)_J2$(J₂).h5", "w") do file
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

    h5open("Data/TDVP/Heisenberg_Dimerized_TEBD_Time$(ttotal)_Delta$(δ)_J2$(J₂).h5", "r+") do file
        write(file, "Psi", ψ)
        write(file, "Sz T=0", Sz₀)
        write(file, "Czz T=0", Czz₀)
        write(file, "Sz Perturbed", Sz₁)
        write(file, "Czz Perturbed", Czz₁)
        write(file, "Czz", Czz)
        write(file, "Czz Odd", Czz_odd) 
        write(file, "Czz Even", Czz_even)
        write(file, "Sz", Sz_all)
        write(file, "Sz Odd", Sz_all_odd)
        write(file, "Sz Even", Sz_all_even)
        write(file, "Bond", chi)
    end

    return
end