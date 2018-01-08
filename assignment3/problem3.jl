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
  ptDirection = indmax(size(points))

  t = sum(points, ptDirection)./size(points,ptDirection);
  (any( i->(i>2),size(t)))?(t=t[1:2]):()

  if ptDirection==1
    s = [norm(points[i,:]) for i=1:size(points,ptDirection)];
  else
    s = [norm(points[:,i]) for i=1:size(points,ptDirection)]
  end
  s = 0.5*maximum(s);

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
  num_p1 = size(p1, indmax(size(p1)))
  num_p2 = size(p2, indmax(size(p2)))
  @assert num_p1==num_p2 ["Num of p1 neq to num of p2!"]


  A = zeros(num_p1,9);
  for i=1:num_p1
    x   = p1[1,i]
    x_d = p2[1,i]
    y   = p1[2,i]
    y_d = p2[2,i]
    A[i,:] = [x*x_d y*x_d x_d x*y_d y*y_d y_d x y 1.0];
  end

  # solve using SVD (thin=false)
  U,S,V = svd(A, thin=false);

  F =       [V[1,9] V[2,9] V[3,9];
             V[4,9] V[5,9] V[6,9];
             V[7,9] V[8,9] V[9,9]];

  # enforce rank of 2
  F = enforcerank2(F);

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
  #F = T2'*Fbar*T1;
  F = T1'*Fbar*T2;

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
  # l1 = F*x2 OR l2 = F' x1
  num_pts = size(points1, indmax(size(points1)))
  pts = Common.cart2hom(points')

  epLine = F * pts
  epLine = Common.hom2cart(epLine)
  @show epLine, size(epLine)

  # e x x
  x = linspace(1,640)

  figure()
  #subplot(211)
  PyPlot.imshow(img)
  for i=1:num_pts
    y = epLine[1,i]*x+epLine[2,i]
    PyPlot.plot(x,y)
  end

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
  # 0 = p1' * F * p2 --> residuals are difference to 0
  # p1 expected in homogenous coords.: 3xN
  num_pts = size(p1, indmax(size(p1)))
  residual = zeros(num_pts,1)

  for i=1:num_pts
    residual[i] =  (p1[:,i]' * F * p2[:,i])[1]
  end

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
