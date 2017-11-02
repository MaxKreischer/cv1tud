using Images  # Basic image processing functions
using PyPlot
using JLD

# Helper functions:
cast2Int(x) = convert(Int64, floor(x))
castMat2Float(A) = convert(Array{Float64,2}, A)
# Load the Bayer image from the provided .jld file
function loadbayer()
  data = JLD.load("./bayerdata.jld", "bayerimg")
  return data::Array{Float64,2}
end


# Separate the Bayer image into three images (one for each color channel), filling up all
# unknown values with 0
function separatebayer(data::Array{Float64,2})
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
  # sources used:
  # https://en.wikipedia.org/wiki/Bayer_filter#Demosaicing ,
  # "Review of Bayer Pattern Color Filter Array (CFA)
  #  Demosaicing with New Quality Assessment Algorithms" by
  #  Robert A. Maschal Jr., S. Susan Young, Joe Reynolds, Keith Krapels,
  #  Jonathan Fanning, and Ted Corbin
  # green correlation kernel:
  nrows, ncols = size(r)
  #kerG  =      castMat2Float(0.25*[0 1 0; 1 4 1; 0 1 0])
  #kerC  =      castMat2Float(0.25*[1 2 1; 2 4 2; 1 2 1])

  #imRed   =   r + Images.imfilter(r, kerNN)
  #imGreen =   g+ Images.imfilter(g, kerNN)
  #imBlue  =   b + Images.imfilter(b, kerNN)
  rFilt = zeros(nrows,ncols)
  gFilt = zeros(nrows,ncols)
  bFilt = zeros(nrows,ncols)

  image = makeimage(imRed, imGreen, imBlue)
  return image::Array{Float64,3}
end


# display two images in a single figure window
function displayimages(img1::Array{Float64,3}, img2::Array{Float64,3})
  # Sources used for plot creation:
  #https://stackoverflow.com/questions/35692507/plot-several-image-files-in-matplotlib-subplots
  #And:
  #https://github.com/gizmaa/Julia_Examples/blob/master/pyplot_subplot.jl
  fig = figure("pyplot_subplot_column")
  subplot(211)
  PyPlot.imshow(img1)
  title("Image with missing values")
  PyPlot.axis("off")
  subplot(212)
  PyPlot.imshow(img2)
  title("Image after filtering")
  PyPlot.axis("off")
  fig[:canvas][:draw]() # Update the figure
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
  nrows, ncols = size(r)
  testZero = zeros(nrows, ncols)
  imgRed = makeimage(r,testZero,testZero)
  imgGreen = makeimage(testZero,g,testZero)
  imgBlue = makeimage(testZero, testZero, b)

  # interpolate Bayer
  #img2 = debayer(r,g,b)
  #PyPlot.imshow(img2)
  # display images
  #displayimages(img1, img1)
  return
end
