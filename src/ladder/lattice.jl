# 1/20/2026
# Implement the lattice geometrey for a ladder system


# """
# 	A LatticeBond is a struct which represents
# 	a single bond in a geometrical lattice or
# 	else on interaction graph defining a physical
# 	model such as a quantum Hamiltonian.

# 	LatticeBond has the following data fields:

# 	- s1::Int -- number of site 1
# 	- s2::Int -- number of site 2
# 	- x1::Float64 -- x coordinate of site 1
# 	- y1::Float64 -- y coordinate of site 1
# 	- x2::Float64 -- x coordinate of site 2
# 	- y2::Float64 -- y coordinate of site 2
# 	- type::String -- optional description of bond type
# """


struct LatticeBond
  s1::Int
  s2::Int
  x1::Float64
  y1::Float64
  x2::Float64
  y2::Float64
  type::String
end

# """
#     LatticeBond(s1::Int,s2::Int)

#     LatticeBond(s1::Int,s2::Int,
#                 x1::Real,y1::Real,
#                 x2::Real,y2::Real,
#                 type::String="")

# 	Construct a LatticeBond struct by
# 	specifying just the numbers of sites
# 	1 and 2, or additional details including
# 	the (x,y) coordinates of the two sites and
# 	an optional type string.
# """


function LatticeBond(s1::Int, s2::Int)
  return LatticeBond(s1, s2, 0.0, 0.0, 0.0, 0.0, "")
end


function LatticeBond(s1::Int, s2::Int, x1::Real, y1::Real, x2::Real, y2::Real, bondtype::String="")
  cf(x) = convert(Float64, x)
  return LatticeBond(s1, s2, cf(x1), cf(y1), cf(x2), cf(y2), bondtype)
end


"""
	Lattice is an alias for Vector{LatticeBond}
"""
const Lattice = Vector{LatticeBond}



# """
#     ladder_lattice(Nx::Int, Ny::Int; kwargs...)::Lattice

#     Return a Lattice (array of LatticeBond
#     objects) corresponding to the ladder 
# 	lattice of dimensions (Nx, Ny).
#     By default the lattice has open boundary along x direction,
# 	and periodic boundary along y direction.
# """

function ladder_lattice(Nx::Int, Ny::Int; yperiodic=true)::Lattice
	"""
		Nx: number of sites along the x direction
		Ny: number of sites along the y direction
	"""

	# Compute the number of sites and bonds
	N = Nx * Ny
	Nbond = 2 * Nx * Ny + Nx - 6


	# Initialize the lattice (a vector of lattice bonds)
  	latt = Lattice(undef, Nbond)
  	b = 0
	for n in 1:N
		x = div(n - 1, Ny) + 1
		y = mod(n - 1, Ny) + 1

		"""
			Set up the nearest neighbor bonds
		"""
		# Nearest-neighbor bond along x direction
		if n <= N - 2
			latt[b += 1] = LatticeBond(n, n + 2)
		end

		# Nearest-neighbor bond along y direction
		if y == 1
			latt[b += 1] = LatticeBond(n, n + 1)
		end


		"""
			Set up the next-nearest neighbor bonds
		"""
		if n <= N - 4
			latt[b += 1] = LatticeBond(n, n + 4)
		end
	end

	if length(latt) != Nbond
		error("\nError in constructing ladder lattice: number of bonds mismatch!")
	end

	# @show latt
	return latt
end