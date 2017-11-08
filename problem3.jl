using Images
using PyPlot
using JLD
using Base.Test

# Helper functions:
cast2Int(x) = convert(Int64, floor(x))
cast2Float(x) = convert(Float64, x)
castMat2Float(A) = convert(Array{Float64,2}, A)
castArr2Float(A) = convert(Array{Float64,1}, A)

# Transform from Cartesian to homogeneous coordinates
function cart2hom(points::Array{Float64,2})
  # P =  p1x p2x p3x . . .
  #      p1y p2y p3y . . .
  #      p1z p2z p3z . . .
  nrows,ncols = size(points)
  points_hom = vcat(points, ones(1, ncols))
  return points_hom::Array{Float64,2}
end



# Transform from homogeneous to Cartesian coordinates
function hom2cart(points::Array{Float64,2})
  nrows, ncols = size(points)
  (nrows>3) ? (dim=3) : (dim=2)
  points_cart = zeros(dim, ncols)
  for col=1:ncols
    points_cart[:, col] = points[1:dim, col]./points[dim+1, col]
  end
  return points_cart::Array{Float64,2}
end


# Translation by v
function gettranslation(v::Array{Float64,1})
  @assert (size(v)==3) "Not a 3-d array of translations."
  T = castMat2Float(diagm([1;1;1;1]))
  for rows=1:3
    T[rows,4] = v[rows]
  end
  return T::Array{Float64,2}
end


# Rotation of d degrees around x axis
function getxrotation(d::Int)
  x = convert(Float64,d)*(pi/180.0)
  Rx = [  1.0 0.0    0.0      0.0;
          0.0 cos(x) -sin(x)  0.0;
          0.0 sin(x) cos(x)   0.0;
          0.0 0.0    0.0      1.0]
  return Rx::Array{Float64,2}
end


# Rotation of d degrees around y axis
function getyrotation(d::Int)
  y = convert(Float64,d)*(pi/180.0)
  Ry = [  cos(y)  0.0 sin(y) 0.0;
          0.0     1.0 0.0    0.0;
          -sin(y) 0.0 cos(y) 0.0;
          0.0     0.0 0.0    1.0]
  return Ry::Array{Float64,2}
end


# Rotation of d degrees around z axis
function getzrotation(d::Int)
  z = convert(Float64,d)*(pi/180.0)
  Rz = [  cos(z) -sin(z) 0.0 0.0;
          sin(z) cos(z)  0.0 0.0;
          0.0    0.0     1.0 0.0;
          0.0    0.0     0.0 1.0]
  return Rz::Array{Float64,2}
end


# Central projection matrix (including camera intrinsics)
function getcentralprojection(principal::Array{Int,1}, focal::Int)
  #
  # (x y 1)'   = K_cam(3x3) * PerspProj(3x4) * World2Cam(4x4) * (X Y Z 1)'
  # central proj -> K*PProj
  f = cast2Float(focal)
  pp =  castArr2Float(principal)
  K =          [f   0.0 pp[1] 0.0;
                0.0 f   pp[2] 0.0;
                0.0 0.0 1.0   0.0]
  return K::Array{Float64,2}
end


# Return full projection matrix P and full model transformation matrix M
function getfullprojection(T::Array{Float64,2},Rx::Array{Float64,2},Ry::Array{Float64,2},Rz::Array{Float64,2},V::Array{Float64,2})
  M = Rz*(Rx*(Ry*T))
  P = V*M
  return P::Array{Float64,2},M::Array{Float64,2}
end


# Load 2D points
function loadpoints()
  points = JLD.load("obj2d.jld", "x")
  return points::Array{Float64,2}
end


# Load z-coordinates
function loadz()
  z = JLD.load("zs.jld", "Z")
  return z::Array{Float64,2}
end


# Invert just the central projection P of 2d points *P2d* with z-coordinates *z*
function invertprojection(P::Array{Float64,2}, P2d::Array{Float64,2}, z::Array{Float64,2})
  # get homogenous P2d then multiply with respective z
  P3d = cart2hom(P2d)
  _,ncols = size(P3d)

  f = 1.0/P[1,1]
  px = P[1,3]
  py = P[2,3]
  iP = [1.0/f 0.0 -px/f; 0.0 1.0/f -py/f; 0.0 0.0 1.0]

  for i=1:ncols
    P3d[:,i] = iP*(P3d[:,i].*z[i])
  end

  return P3d::Array{Float64,2}
end


# Invert just the model transformation of the 3D points *P3d*
function inverttransformation(A::Array{Float64,2}, P3d::Array{Float64,2})
  #Inverse of homogenous transformation given by:
  # M^-1 = [R^-1 -R^-1 *d ; 0 0 0 1]; from:
  # https://mathematica.stackexchange.com/questions/106257/how-do-i-get-the-inverse-of-a-homogeneous-transformation-matrix
  X = cart2hom(P3d)
  _,ncols = size(X)
  iRot = (A[1:3, 1:3])'
  t = A[1:3, 4]
  iTrans = [ iRot -iRot*t; 0.0 0.0 0.0 1.0 ]
  for i=1:ncols
    X[:,i] = iTrans*X[:,i]
  end
  return X::Array{Float64,2}
end


# Plot 2D points
function displaypoints2d(points::Array{Float64,2})

  return gcf()::Figure
end


# Plot 3D points
function displaypoints3d(points::Array{Float64,2})

  return gcf()::Figure
end


# Apply full projection matrix *C* to 3D points *X*
function projectpoints(P::Array{Float64,2}, X::Array{Float64,2})
  P2d = cart2hom(X)
  _,ncols = size(P2d)
  for i=1:ncols
    P2d[:,i] = P*P2d[:,i]
  end
  P2d = hom2cart(P2d)
  return P2d:::Array{Float64,2}
end


#= Problem 3
Projective Transformation =#


function problem3()
  # parameters
  t               = [-27.1; -2.9; -3.2]
  principal_point = [8; -10]
  focal_length    = 8

  # model transformations
  T = gettranslation(t)
  Ry = getyrotation(135)
  Rx = getxrotation(-30)
  Rz = getzrotation(90)

  # central projection including camera intrinsics
  K = getcentralprojection(principal_point,focal_length)

  # full projection and model matrix
  P,M = getfullprojection(T,Rx,Ry,Rz,K)

  # load data and plot it
  points = loadpoints()
  displaypoints2d(points)

  # reconstruct 3d scene
  z = loadz()
  Xt = invertprojection(K,points,z)
  Xh = inverttransformation(M,Xt)
  worldpoints = hom2cart(Xh)
  displaypoints3d(worldpoints)

  # reproject points
  points2 = projectpoints(P,worldpoints)
  displaypoints2d(points2)

  @test_approx_eq points points2
  return
end
