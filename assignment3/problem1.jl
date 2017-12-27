using Images
using PyPlot

include("Common.jl")

#---------------------------------------------------------
# Loads grayscale and color image given PNG filename.
#
# INPUTS:
#   filename     given PNG image file
#
# OUTPUTS:
#   gray         single precision grayscale image
#   rgb          single precision color image
#
#---------------------------------------------------------
function loadimage(filename)
  rgb = imread(filename);
  gray = Common.rgb2gray(rgb);
  return gray::Array{Float32,2}, rgb::Array{Float32,3}
end


#---------------------------------------------------------
# Computes structure tensor.
#
# INPUTS:
#   img             grayscale color image
#   sigma           std for presmoothing derivatives
#   sigma_tilde     std for presmoothing coefficients
#   fsize           filter size to use for presmoothing
#
# OUTPUTS:
#   S_xx       first diagonal coefficient of structure tensor
#   S_yy       second diagonal coefficient of structure tensor
#   S_xy       off diagonal coefficient of structure tensor
#
#---------------------------------------------------------
function computetensor(img::Array{Float64,2},sigma::Float64,sigma_tilde::Float64,fsize::Int)
  gaussDerivs = Common.gauss2d(sigma, [fsize, fsize]);
  gaussCoeffs = Common.gauss2d(sigma_tilde, [fsize, fsize]);
  x_deriv = [-1.0, 0.0, 1.0]';
  y_deriv = x_deriv';
  imSmooth = imfilter(img, gaussDerivs);
  imDx = imfilter(imSmooth, x_deriv);
  imDy = imfilter(imSmooth, y_deriv);
  S_xx = imDx.^2.0;
  S_yy = imDy.^2.0;
  S_xy = imDx.*imDy;
  S_xx = imfilter(S_xx, gaussCoeffs);
  S_yy = imfilter(S_yy, gaussCoeffs);
  S_xy = imfilter(S_xy, gaussCoeffs);
  return S_xx::Array{Float64,2},S_yy::Array{Float64,2},S_xy::Array{Float64,2}
end


#---------------------------------------------------------
# Computes Harris function values.
#
# INPUTS:
#   S_xx       first diagonal coefficient of structure tensor
#   S_yy       second diagonal coefficient of structure tensor
#   S_xy       off diagonal coefficient of structure tensor
#   sigma      std that was used for presmoothing derivatives
#   alpha      weighting factor for trace
#
# OUTPUTS:
#   harris     Harris function score
#
#---------------------------------------------------------
function computeharris(S_xx::Array{Float64,2},S_yy::Array{Float64,2},S_xy::Array{Float64,2}, sigma::Float64, alpha::Float64)
  nrows, ncols = size(S_xx);
  harris = zeros(nrows,ncols);
  for i=1:nrows
    for j=1:ncols
      S = [S_xx[i,j] S_xy[i,j]; S_xy[i,j] S_yy[i,j]];
      harris[i,j] = sigma.^4.0*(det(S) - alpha*(trace(S)).^2.0);
    end
  end
  return harris::Array{Float64,2}
end


#---------------------------------------------------------
# Non-maximum suppression of Harris function values.
#   Extracts local maxima within a 5x5 stencils.
#   Allows multiple points with equal values within the same window.
#   Applies thresholding with the given threshold.
#
# INPUTS:
#   harris     Harris function score
#   thresh     param for thresholding Harris function
#
# OUTPUTS:
#   px        x-position of kept Harris interest points
#   py        y-position of kept Harris interest points
#
#---------------------------------------------------------
function nonmaxsupp(harris::Array{Float64,2}, thresh::Float64)
  nrows,ncols = size(harris);
  harris_nonmax = Common.nlfilter(harris, maximum, 5, 5);
  harris_nonmax[1:2,:] = zeros(size(harris_nonmax[1:2,:]));
  harris_nonmax[end:-1:end-1,:] = zeros(size(harris_nonmax[end:-1:end-1,:]));
  harris_nonmax[:,1:2] = zeros(size(harris_nonmax[:,1:2]));
  harris_nonmax[:,end:-1:end-1] = zeros(size(harris_nonmax[:,end:-1:end-1]));
  for i=1:nrows
    for j=1:ncols
      (harris_nonmax[i,j]<thresh)?(harris_nonmax[i,j] = 0.0):();
    end
  end
  px = Array{Int64,1}();
  py = Array{Int64,1}();
  # rows -> y, cols -> x
  for row=1:nrows
    for col=1:ncols
      if(harris_nonmax[row,col] != 0.0)
        push!(px, col);
        push!(py, row);
      end
    end
  end
  return px::Array{Int,1},py::Array{Int,1}
end


#---------------------------------------------------------
# Problem 1: Harris Detector
#---------------------------------------------------------
function problem1()
  # parameters
  sigma = 2.4               # std for presmoothing derivatives
  sigma_tilde = 1.6*sigma   # std for presmoothing coefficients
  fsize = 25                # filter size for presmoothing
  alpha = 0.06              # Harris alpha
  threshold = 1e-7          # Harris function threshold

  # Load both colored and grayscale image from PNG file
  gray,rgb = loadimage("a3p1.png")
  # Convert to double precision
  gray = Float64.(gray)
  rgb = Float64.(rgb)

  # Compute the three coefficients of the structure tensor
  S_xx,S_yy,S_xy = computetensor(gray,sigma,sigma_tilde,fsize)

  # Compute Harris function value
  harris = computeharris(S_xx,S_yy,S_xy,sigma,alpha)

  # Display Harris images
  figure()
  imshow(harris,"jet",interpolation="none")
  axis("off")
  title("Harris function values")

  # Threshold Harris function values
  mask = harris .> threshold
  y,x = findn(mask)
  figure()
  imshow(rgb)
  plot(x,y,"xy")
  axis("off")
  title("Harris interest points without non-maximum suppression")
  gcf()

  # Apply non-maximum suppression
  x,y = nonmaxsupp(harris,threshold)

  # Display interest points on top of color image
  figure()
  imshow(rgb)
  plot(x,y,"xy",linewidth=8)
  axis("off")
  title("Harris interest points after non-maximum suppression")
  return nothing
end

function showcrap(img,x1,y1,x2,y2)
  figure()
  subplot(121)
  imshow(img)
  plot(x1,y1, "xy")
  axis("off")
  subplot(122)
  imshow(img)
  plot(x2,y2, "xy")
  axis("off")
  gcf()
end


#   Extracts local maxima within a 5x5 stencils.
#   Allows multiple points with equal values within the same window.
#   Applies thresholding with the given threshold.
function test(x)
  return maximum(x)
end
