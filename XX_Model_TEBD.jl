# 07/25/2024
# Use time evolving block decimation (TEBD) to simulate the time evolution of a 1D XX model.


using ITensors, ITensorMPS
using HDF5

let 
    N = 200
    cutoff = 1E-10
    τ = 0.1
    ttotal = 100.0

    # Make an array of "site" indices
    s = siteinds("S=1/2", N; conserve_qns=true)

    # Make gates (1, 2), (2, 3), ..., (N-1, N)
    gates = ITensor[]
    for index in 1 : N - 1
        s₁ = s[index]
        s₂ = s[index + 1]
        hj = 1/2 * op("S+", s₁) * op("S-", s₂) + 1/2 * op("S-", s₁) * op("S+", s₂)
        Gj = exp(-im * τ/2 * hj)
        push!(gates, Gj)
    end
    append!(gates, reverse(gates))

    # Run DMRG simulation to obtain the ground-state wave function
    os = OpSum()
    for index = 1 : N - 1
        os += 1/2, "S+", index, "S-", index + 1
        os += 1/2, "S-", index, "S+", index + 1
    end
    Hamiltonian = MPO(os, s)
    ψ₀ = MPS(s, n -> isodd(n) ? "Up" : "Dn")
    

    # Define parameters that are used in the DMRG optimization process
    nsweeps = 20
    maxdim = [20, 50, 200, 1000]
    states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    # ψ = randomMPS(s, states; linkdims = 10)
    E, ψ = dmrg(Hamiltonian, ψ₀; nsweeps, maxdim, cutoff)
    
    Sz₀ = expect(ψ, "Sz"; sites = 50 : 51)
    Czz₀ = correlation_matrix(ψ, "Sz", "Sz"; sites = 50 : 51)
    @show Sz₀
    @show Czz₀
    
    center = div(N, 2)
    ψ_copy = deepcopy(ψ)
    
    
    
    # Apply a local operator Sz to the center of the chain
    local_op = op("Sz", s[center])
    ψ = apply(local_op, ψ; cutoff)  
    normalize!(ψ)

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
   
    Sz₁ = expect(ψ, "Sz"; sites = 50 : 51)
    Czz₁ = correlation_matrix(ψ, "Sz", "Sz"; sites = 50 : 51)
    @show Sz₁
    @show Czz₁

    Czz = Matrix{ComplexF64}(undef, Int(ttotal / τ), N * N)
    Czz_unequaltime = Matrix{ComplexF64}(undef, Int(ttotal / τ), N)
    @show size(Czz_unequaltime)
    chi = Matrix{Float64}(undef, Int(ttotal / τ), N - 1)
    @show size(chi)
    Sz_all = Matrix{ComplexF64}(undef, Int(ttotal / τ), N)


    # Time evovle the original and perturbed wave functions
    for t in 0 : τ : ttotal
        index = round(Int, t / τ) + 1
        @show index
        Sz = expect(ψ, "Sz"; sites = center)
        println("t = $t, Sz = $Sz")

        t ≈ ttotal && break
        ψ = apply(gates, ψ; cutoff)
        normalize!(ψ)
        chi[index, :] = linkdims(ψ)
        @show linkdims(ψ)

        ψ_copy = apply(gates, ψ_copy; cutoff)
        normalize!(ψ_copy)

        Czz[index, :] = correlation_matrix(ψ, "Sz", "Sz"; sites = 1 : N)
        Sz_all[index, :] = expect(ψ, "Sz"; sites = 1 : N)

        for site_index in collect(1 : N)
            tmp_os = OpSum()
            tmp_os += "Sz", site_index
            tmp_MPO = MPO(tmp_os, s)
            Czz_unequaltime[index, site_index] = inner(ψ_copy', tmp_MPO, ψ)
        end
    end
    # @show Czz_unequaltime

    
    h5open("Data/XX_Model_TEBD_N$(N)_Time$(ttotal)_tau$(τ).h5", "w") do file
        write(file, "Psi", ψ)
        write(file, "Czz_unequaltime", Czz_unequaltime)
        write(file, "Czz", Czz)
        write(file, "Sz", Sz_all)
        write(file, "Bond", chi)
    end

    return
end