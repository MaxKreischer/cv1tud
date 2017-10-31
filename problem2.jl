using Images  # Basic image processing functions
using PyPlot
using JLD


# Load the Bayer image from the provided .jld file
function loadbayer()
  data = JLD.load("./bayerdata.jld", "bayerimg")
  return data::Array{Float64,2}
end


# Separate the Bayer image into three images (one for each color channel), filling up all
# unknown values with 0
function separatebayer(data::Array{Float64,2})
  # Helper functions:
  cast2Int(x) = convert(Int64, floor(x))
  castMat2Float(A) = convert(Array{Float64,2}, A)
  # From assigmnment PDF:
  # G R G R
  # B G B G
  # G R G R
  # B G B G
  red   =   castMat2Float([0 1 0 1; 0 0 0 0; 0 1 0 1; 0 0 0 0])
  green =   castMat2Float([1 0 1 0; 0 1 0 1; 1 0 1 0; 0 1 0 1])
  blue  =   castMat2Float([0 0 0 0; 1 0 1 0; 0 0 0 0; 1 0 1 0])
  nrows, ncols = size(data)
  rMask = repmat(red,   cast2Int(nrows/4),cast2Int(ncols/4))
  gMask = repmat(green, cast2Int(nrows/4),cast2Int(ncols/4))
  bMask = repmat(blue,  cast2Int(nrows/4),cast2Int(ncols/4))
  r = data .* rMask
  g = data .* gMask
  b = data .* bMask
  return r::Array{Float64,2}, g::Array{Float64,2}, b::Array{Float64,2}
end


# Combine three color channels into a single image
function makeimage(r::Array{Float64,2}, g::Array{Float64,2}, b::Array{Float64,2})
  nrows, ncols = size(r)
  image = Array{Float64}(nrows, ncols, 3)
  image[:,:,1] = r[:,:]
  image[:,:,2] = g[:,:]
  image[:,:,3] = b[:,:]
  return image::Array{Float64,3}
end


# Interpolate missing color values using bilinear interpolation
function debayer(r::Array{Float64,2}, g::Array{Float64,2}, b::Array{Float64,2})

  return image::Array{Float64,3}
end


# display two images in a single figure window
function displayimages(img1::Array{Float64,3}, img2::Array{Float64,3})

  # you may reuse your function from problem1 here
  return nothing
end


#= Problem 2
Bayer Interpolation =#

function problem2()
  # load raw data
  data = loadbayer()

  # separate data
  r,g,b = separatebayer(data)

  # merge raw Bayer
  img1 = makeimage(r,g,b)
  PyPlot.imshow(img1)
  # interpolate Bayer
  #img2 = debayer(r,g,b)

  # display images
  #displayimages(img1, img2)
  #return
end
