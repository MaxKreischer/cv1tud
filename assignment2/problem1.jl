using Images
using PyPlot

F2I(x) = convert(Int64, x);
# Create a gaussian filter
function makegaussianfilter(size::Array{Int,2},sigma::Float64)
  gauss(x, sigma) = exp(-(x^2.0/(2*(sigma^2.0))));
  nrows = size[1];
  ncols = size[2];
  filterX = zeros(nrows,1);
  filterY = zeros(1,ncols);
  if(nrows%2==0)
    throw("NOT IMPLEMENTED YET! -> makegaussianfilter for even filters.")
  else
    # X direction
    #liX = linspace(0,3*sigma,ceil(nrows/2))
    #filterX_ = map(g, liX);
    filterX_ = [(1.0/sqrt(2.0*pi)*sigma)*exp(-(x^2.0/(2*(sigma^2.0)))) for x in linspace(0, 3*sigma, ceil(nrows/2))];
    halfX = convert(Int64, ceil(nrows/2));
    filterX[1:halfX-1] = copy(filterX_[end:-1:2]);
    filterX[halfX] = copy(filterX_[1]);
    filterX[halfX+1:end] = copy(filterX_[2:end]);
    # Y direction
    #liY = linspace(0,3*sigma,ceil(ncols/2))
    #filterY_ = map(g, liY);
    filterY_ = [(1.0/sqrt(2.0*pi)*sigma)*exp(-(y^2.0/(2*(sigma^2.0)))) for y in linspace(0, 3*sigma, ceil(ncols/2))];
    halfY = convert(Int64, ceil(ncols/2));
    filterY[1:halfY-1] = copy(filterY_[end:-1:2]);
    filterY[halfY] = copy(filterY_[1]);
    filterY[halfY+1:end] = copy(filterY_[2:end]);
  end
  f = filterX * filterY;
  #f = copy(f ./ sum(f));
  return f::Array{Float64,2}
end

# Create a binomial filter1
function makebinomialfilter(size::Array{Int,2})
  n_pick_k(N,k) = factorial(N)/(factorial(k)*factorial(N-k));
  nrows = size[1]-1;
  ncols = size[2]-1;
  # X
  weightsX = [n_pick_k(nrows, i) for i=0:nrows];
  # Y
  weightsY = [n_pick_k(ncols, i) for i=0:ncols];
  f = weightsX'' * weightsY';
  f = f ./ sum(f);
  return f::Array{Float64,2}
end

# Downsample an image by a factor of 2
function downsample2(A::Array{Float64,2})
  nrows,ncols = size(A);
  rowIdx = 1:2:nrows;
  colIdx = 1:2:ncols;
  D = [A[i,j] for i in rowIdx for j in colIdx];
  D = reshape(D, (F2I(ceil(nrows/2)), F2I(ceil(ncols/2))))'
  return D::Array{Float64,2}
end

# Upsample an image by a factor of 2
function upsample2(A::Array{Float64,2},fsize::Array{Int,2})

  return U::Array{Float64,2}
end

# Build a gaussian pyramid from an image.
# The output array should contain the pyramid levels in decreasing sizes.
function makegaussianpyramid(im::Array{Float32,2},nlevels::Int,fsize::Array{Int,2},sigma::Float64)

  return G::Array{Array{Float64,2},1}
end

# Display a given image pyramid (laplacian or gaussian)
function displaypyramid(P::Array{Array{Float64,2},1})

  return nothing::Void
end

# Build a laplacian pyramid from a gaussian pyramid.
# The output array should contain the pyramid levels in decreasing sizes.
function makelaplacianpyramid(G::Array{Array{Float64,2},1},nlevels::Int,fsize::Array{Int,2})

  return L::Array{Array{Float64,2},1}
end

# Amplify frequencies of the first two layers of the laplacian pyramid
function amplifyhighfreq2(L::Array{Array{Float64,2},1})

  return A::Array{Array{Float64,2},1}
end

# Reconstruct an image from the laplacian pyramid
function reconstructlaplacianpyramid(L::Array{Array{Float64,2},1},fsize::Array{Int,2})

  return im::Array{Float64,2}
end


# Problem 1: Image Pyramids and Image Sharpening

function problem1()
  # parameters
  fsize = [5 5]
  sigma = 1.5
  nlevels = 6

  # load image
  im = PyPlot.imread("./data-julia/a2p1.png")

  # create gaussian pyramid
  G = makegaussianpyramid(im,nlevels,fsize,sigma)

  return "Test RET"


  # display gaussianpyramid
  displaypyramid(G)
  title("Gaussian Pyramid")

  # create laplacian pyramid
  L = makelaplacianpyramid(G,nlevels,fsize)

  # dispaly laplacian pyramid
  displaypyramid(L)
  title("Laplacian Pyramid")

  # amplify finest 2 subands
  L_amp = amplifyhighfreq2(L)

  # reconstruct image from laplacian pyramid
  im_rec = reconstructlaplacianpyramid(L_amp,fsize)

  # display original and reconstructed image
  figure()
  subplot(131)
  imshow(im,"gray",interpolation="none")
  axis("off")
  title("Original Image")
  subplot(132)
  imshow(im_rec,"gray",interpolation="none")
  axis("off")
  title("Reconstructed Image")
  subplot(133)
  imshow(im-im_rec,"gray",interpolation="none")
  axis("off")
  title("Difference")
  gcf()

  return
end
