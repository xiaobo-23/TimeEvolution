## 08/23/2024
## 

using ITensors: MPO, OpSum, dmrg, inner, random_mps, siteinds
using ITensorTDVP: tdvp

function main()
    n = 10
    s = siteinds("S=1/2", n)
  
    function heisenberg(n; J1 = 1.0, J2 = 0.5)
        os = OpSum()

        if !iszero(J1)
            for j in 1:(n - 1)
                os += J1 / 2, "S+", j, "S-", j + 1
                os += J1 / 2, "S-", j, "S+", j + 1
                os += J1, "Sz", j, "Sz", j + 1
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
    H = MPO(heisenberg(n; J1, J2), s)
    ψ = random_mps(s, "↑"; linkdims=10)
  
    @show inner(ψ', H, ψ) / inner(ψ, ψ)
  
    ϕ = tdvp(
      H,
      -20.0,
      ψ;
      time_step=-1.0,
      maxdim=30,
      cutoff=1e-10,
      normalize=true,
      reverse_step=false,
      outputlevel=1,
    )
    @show inner(ϕ', H, ϕ) / inner(ϕ, ϕ)
  
    e2, ϕ2 = dmrg(H, ψ; nsweeps=10, maxdim=20, cutoff=1e-10)
    @show inner(ϕ2', H, ϕ2) / inner(ϕ2, ϕ2), e2
  
    return nothing
  end
  
  main()