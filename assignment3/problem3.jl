using Images
using PyPlot
using JLD

include("Common.jl")


#---------------------------------------------------------
# Conditioning: Normalization of coordinates for numeric stability.
#
# INPUTS:
#   points    unnormalized coordinates
#
# OUTPUTS:
#   U         normalized (conditioned) coordinates
#   T         [3x3] transformation matrix that is used for
#                   conditioning
#
#---------------------------------------------------------
function condition(points::Array{Float64,2})
  # just insert your problem2 condition method here..
  U = zeros(size(points));
  t = zeros(2,1);
  vec = [points[i,:] for i in 1:size(points,1)];
  s = 0.5*maximum(map(norm,vec));
  t[1] = mean(points[:,1]);
  t[2] = mean(points[:,2]);
  T = [1.0/s 0.0 -t[1]/s; 0.0 1.0/s -t[2]/s; 0.0 0.0 1.0];
  U = T*points;

  @assert size(U) == size(points)
  @assert size(T) == (3,3)
  return U::Array{Float64,2},T::Array{Float64,2}
end


#---------------------------------------------------------
# Enforce a rank of 2 to a given 3x3 matrix.
#
# INPUTS:
#   A     [3x3] matrix (of rank 3)
#
# OUTPUTS:
#   Ahat  [3x3] matrix of rank 2
#
#---------------------------------------------------------
# Enforce that the given matrix has rank 2
function enforcerank2(A::Array{Float64,2})
  U,S,V = svd(A, thin=false);
  S[3] = 0.0;
  Ahat = U*diagm(S)*V';

  @assert size(Ahat) == (3,3)
  return Ahat::Array{Float64,2}
end


#---------------------------------------------------------
# Compute fundamental matrix from conditioned coordinates.
#
# INPUTS:
#   p1     set of conditioned coordinates in left image
#   p2     set of conditioned coordinates in right image
#
# OUTPUTS:
#   F      estimated [3x3] fundamental matrix
#
#---------------------------------------------------------
# Compute the fundamental matrix for given conditioned points
function computefundamental(p1::Array{Float64,2},p2::Array{Float64,2})
  # build homog. lin. eq. system
  # TODO: draw 8pts at random or just take whatever pts?
  A = zeros(8,9);
  for i=1:8
    A[i,:] = reshape(p2[:,i]*p1[:,i]',(1,9));
  end

  # solve using SVD (thin=false)
  U,S,V = svd(A, thin=false);

  Ftilde = [V[1,end] V[2,end] V[3,end];
            V[4,end] V[5,end] V[6,end];
            V[7,end] V[8,end] V[9,end]];

  # enforce rank of 2
  F = enforcerank2(Ftilde);

  @assert size(F) == (3,3)
  return F::Array{Float64,2}
end


#---------------------------------------------------------
# Compute fundamental matrix from unconditioned coordinates.
#
# INPUTS:
#   p1     set of unconditioned coordinates in left image
#   p2     set of unconditioned coordinates in right image
#
# OUTPUTS:
#   F      estimated [3x3] fundamental matrix
#
#---------------------------------------------------------
function eightpoint(p1::Array{Float64,2},p2::Array{Float64,2})
  p1cond, T1 = condition(p1);
  p2cond, T2 = condition(p2);
  Fbar = computefundamental(p1cond,p2cond);
  F = T2'*Fbar*T1;


  @assert size(F) == (3,3)
  return F::Array{Float64,2}
end


#---------------------------------------------------------
# Draw epipolar lines:
#   E.g. for a given fundamental matrix and points in first image,
#   draw corresponding epipolar lines in second image.
#
#
# INPUTS:
#   Either:
#     F         [3x3] fundamental matrix
#     points    set of coordinates in left image
#     img       right image to be drawn on
#
#   Or:
#     F         [3x3] transposed fundamental matrix
#     points    set of coordinates in right image
#     img       left image to be drawn on
#
#---------------------------------------------------------
function showepipolar(F::Array{Float64,2},points::Array{Float64,2},img::Array{Float64,3})





  return nothing::Void
end


#---------------------------------------------------------
# Compute the residual errors for a given fundamental matrix F,
# and set of corresponding points.
#
# INPUTS:
#    p1    corresponding points in left image
#    p2    corresponding points in right image
#    F     [3x3] fundamental matrix
#
# OUTPUTS:
#   residuals      residual errors for given fundamental matrix
#
#---------------------------------------------------------
function computeresidual(p1::Array{Float64,2},p2::Array{Float64,2},F::Array{Float64,2})





  return residual::Array{Float64,2}
end



#---------------------------------------------------------
# Problem 3: Fundamental Matrix
#---------------------------------------------------------
function problem3()
  # Load images and points
  img1 = Float64.(PyPlot.imread("a3p3a.png"))
  img2 = Float64.(PyPlot.imread("a3p3b.png"))
  points1 = load("points.jld", "points1")
  points2 = load("points.jld", "points2")

  # Display images and correspondences
  figure()
  subplot(121)
  imshow(img1,interpolation="none")
  axis("off")
  scatter(points1[:,1],points1[:,2])
  title("Keypoints in left image")
  subplot(122)
  imshow(img2,interpolation="none")
  axis("off")
  scatter(points2[:,1],points2[:,2])
  title("Keypoints in right image")

  # compute fundamental matrix with homogeneous coordinates
  x1 = Common.cart2hom(points1')
  x2 = Common.cart2hom(points2')
  F = eightpoint(x1,x2)

  # draw epipolar lines
  figure()
  subplot(121)
  showepipolar(F',points2,img1)
  scatter(points1[:,1],points1[:,2])
  title("Epipolar lines in left image")
  subplot(122)
  showepipolar(F,points1,img2)
  scatter(points2[:,1],points2[:,2])
  title("Epipolar lines in right image")

  # check epipolar constraint by computing the remaining residuals
  residual = computeresidual(x1, x2, F)
  println("Residuals:")
  println(residual)

  # compute epipoles
  U,_,V = svd(F)
  e1 = V[1:2,3]./V[3,3]
  println("Epipole 1: $(e1)")
  e2 = U[1:2,3]./U[3,3]
  println("Epipole 2: $(e2)")

  return
end
