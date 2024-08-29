## 08/23/2024
## 
using ITensors
using ITensors: MPO, OpSum, dmrg, inner, random_mps, siteinds
using ITensorTDVP: tdvp
using Observers: observer
using HDF5

function main()
    n = 200
    ttotal = -1.0im
    s = siteinds("S=1/2", n)

  
    function heisenberg(n; J1 = 1.0, J2 = 0.5, Δ = 0.2)
        os = OpSum()

        if !iszero(J1)
            for j in 1:2:(n - 1)
                # @show j
                os += J1 * (1 + Δ) / 2, "S+", j, "S-", j + 1
                os += J1 * (1 + Δ) / 2, "S-", j, "S+", j + 1
                os += J1 * (1 + Δ), "Sz", j, "Sz", j + 1
            end
            
            for j in 2:2:(n - 2)
                # @show j
                os += J1 * (1 - Δ) / 2, "S+", j, "S-", j + 1
                os += J1 * (1 - Δ) / 2, "S-", j, "S+", j + 1
                os += J1 * (1 - Δ), "Sz", j, "Sz", j + 1
            end
        end

        if !iszero(J2)
            for j in 1:(n - 2)
                os += J2 / 2, "S+", j, "S-", j + 2
                os += J2 / 2, "S-", j, "S+", j + 2
                os += J2, "Sz", j, "Sz", j + 2
            end
        end
        return os
    end
  
    J1 = 1.0 
    J2 = 0.5
    Δ = 0.2
    H = MPO(heisenberg(n; J1, J2, Δ), s)
    ψ = random_mps(s, "↑"; linkdims=10)
    @show inner(ψ', H, ψ) / inner(ψ, ψ)
  

    # Obtain the ground state using DMRG
    # ϕ0 = random_mps(s, "↑"; linkdims=10)
    e0, ϕ0 = dmrg(H, ψ; nsweeps=10, maxdim=100, cutoff=1e-10)
    @show inner(ϕ0', H, ϕ0) / inner(ϕ0, ϕ0)

    
    # Apply a local perturbation Sz onto the even site in the center of the chain
    center_even = div(n, 2)
    ϕ_even = deepcopy(ϕ0)
    local_op = op("Sz", s[center_even])
    ϕ_even = apply(local_op, ϕ_even; cutoff)  
    # normalize!(ϕ_even)


    # Apply a local perturbation Sz onto the odd site in the center of the chain  
    center_odd  = div(n, 2) - 1
    ϕ_odd = deepcopy(ϕ0)
    local_operator_odd = op("Sz", s[center_odd])
    ϕ_odd = apply(local_operator_odd, ϕ_odd; cutoff)
    # normalize!(ϕ_odd)


    ϕ = tdvp(
      H,
      ttotal,
      ϕ0;
    #   time_step=-1.0,
      nsweeps=10,
      maxdim=100,
      cutoff=1e-10,
      normalize=true,
      reverse_step=false,
      outputlevel=1,
    )
    Sz = expect(ϕ, "Sz"; sites=1:n)
    Czz = correlation_matrix(ϕ, "Sz", "Sz"; sites=1:n)
    @show inner(ϕ', H, ϕ) / inner(ϕ, ϕ)
    
   
    # Time evolve the wave function perturbed in the center of the chain: odd siteinds
    ϕ_odd_time = tdvp(
      H,
      ttotal,
      ϕ_odd;
    #   time_step=-1.0,
      nsweeps=10,
      maxdim=100,
      cutoff=1e-10,
      normalize=true,
      reverse_step=false,
      outputlevel=1,
    )
    Sz_odd = expect(ϕ_odd_time, "Sz"; sites=1:n)
    Czz_odd = correlation_matrix(ϕ_odd_time, "Sz", "Sz"; sites=1:n)
    @show inner(ϕ_odd_time', H, ϕ_odd_time) / inner(ϕ_odd_time, ϕ_odd_time)


    # Time evolve the wave fcuntion perturbed in the center of the chain: even site 
    ϕ_even_time = tdvp(
      H,
      ttotal,
      ϕ_even;
    #.   time_step=-1.0,
      nsweeps=10,
      maxdim=100,
      cutoff=1e-10,
      normalize=true,
      reverse_step=false,
      outputlevel=1,
    )
    Sz_even = expect(ϕ_even_time, "Sz"; sites=1:n)
    Czz_even = correlation_matrix(ϕ_even_time, "Sz", "Sz"; sites=1:n)
    @show inner(ϕ_even_time', H, ϕ_even_time) / inner(ϕ_even_time, ϕ_even_time)

    
    # Compute unequal-time spin correlation function at the final time
    Czz_unequaltime_odd  = Matrix{ComplexF64}(undef, 1, n)
    Czz_unequaltime_even = Matrix{ComplexF64}(undef, 1, n)

    for index in collect(1 : n)
        tmp_os = OpSum()
        tmp_os += "Sz", index
        tmp_MPO = MPO(tmp_os, s)
        Czz_unequaltime_odd[index]  = inner(ϕ0', tmp_MPO, ϕ_odd_time)
        Czz_unequaltime_even[index] = inner(ϕ0', tmp_MPO, ϕ_even_time) 
    end


    # # Obtain the ground-state energy through DMRG
    # e2, ϕ2 = dmrg(H, ψ; nsweeps=10, maxdim=100, cutoff=1e-10)
    # @show inner(ϕ2', H, ϕ2) / inner(ϕ2, ϕ2), e2

    
    # Save the wave function and observables
    h5open("Data/Heisenberg_Dimerized_TDVP_N$(n)_Real$(real(ttotal))_Imag$(imag(ttotal))_J2$(J2)_Delta$(Δ).h5", "w") do file
        write(file, "Sz", Sz)
        write(file, "Sz_odd", Sz_odd)
        write(file, "Sz_even", Sz_even)
        write(file, "Czz", Czz)
        write(file, "Czz_odd", Czz_odd)
        write(file, "Czz_even", Czz_even)
        write(file, "Czz_unequaltime_odd", Czz_unequaltime_odd)
        write(file, "Czz_unequaltime_even", Czz_unequaltime_even)
    end

    return nothing
  end
  
  main()