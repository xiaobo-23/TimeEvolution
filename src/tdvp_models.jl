## 08/23/2024
## Develop time dependent variational principle (TDVP) to simulate the time evolution of spin models and compare with TEBD results 
## Designed to simulate neutron scattering experiments and quantum dynamics for the NEAT LDRD 

using ITensors
using ITensorMPS
using Observers: observer
using HDF5


# Set up the dimmerized J₁-J₂ Heisenberg Hamiltonian using OpSum
function heisenberg_dimerized(n; J1 = 1.0, J2 = 0.0, Δ = 0.0)
    println(repeat("#", 200))
    println("Setting up the Heisenberg Hamiltonian using OpSum...")
    @show n, J1, J2, Δ
    
    # Initialize OpSum
    os = OpSum()

    # Set up the nearest-neighbor interaction with dimerization
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

    # Set up the next-nearest-neighbor interaction 
    if !iszero(J2)
        for j in 1:(n - 2)
            os += J2 / 2, "S+", j, "S-", j + 2
            os += J2 / 2, "S-", j, "S+", j + 2
            os += J2, "Sz", j, "Sz", j + 2
        end
    end
    return os
end



# Set up the dimmerized J₁-J₂ Heisenberg Hamiltonian using OpSum
function heisenberg(n; J1 = 1.0, J2 = 0.0)
    println(repeat("#", 200))
    println("Setting up the Heisenberg Hamiltonian using OpSum...")
    @show n, J1, J2

    # Initialize OpSum
    os = OpSum()

    # Set up the nearest-neighbor interaction with dimerization
    if !iszero(J1)
        for j in 1:(n-1)
          os += J1/2, "S+", j, "S-", j+1
          os += J1/2, "S-", j, "S+", j+1  
          os += J1, "Sz", j, "Sz", j+1
        end
    end

    # Set up the next-nearest-neighbor interaction 
    if !iszero(J2)
        for j in 1:(n - 2)
            os += J2 / 2, "S+", j, "S-", j + 2
            os += J2 / 2, "S-", j, "S+", j + 2
            os += J2, "Sz", j, "Sz", j + 2
        end
    end

    return os
end