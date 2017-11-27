using Images
using PyPlot

F2I(x) = convert(Int64, x);
# Create a gaussian filter
function makegaussianfilter(size::Array{Int,2},sigma::Float64)
  gauss(x, std=sigma) = (1.0/(sqrt(2.0*pi)*std))*exp((-x.^2)/(2*std.^2));
  # Resource allocation
  x_size = size[1];
  y_size = size[2];
  upperS = 4.5;
  lowerS = -upperS;
  suppSize = 200;
  support = linspace(lowerS,upperS,suppSize);
  # X
  weights_x = zeros(x_size);
  chunks_x = [k for k=1:floor(Int64,suppSize/x_size):suppSize];
  len = Base.size(chunks_x)[1];
  (len==x_size)?(push!(chunks_x,suppSize)):();
  for i =1:endof(chunks_x)
    left =  chunks_x[i];
    (i!=endof(chunks_x))?(right = chunks_x[i+1]):(break);
    weights_x[i] = sum(gauss(support[left:right]));
  end
  # Y
  weights_y = zeros(y_size);
  chunks_y = [k for k=1:floor(Int64,suppSize/y_size):suppSize];
  len = Base.size(chunks_y)[1];
  (len==y_size)?(push!(chunks_y,suppSize)):();
  for i =1:endof(chunks_y)
    left =  chunks_y[i];
    (i!=endof(chunks_y))?(right = chunks_y[i+1]):(break);
    weights_y[i] = sum(gauss(support[left:right]));
  end
  f = weights_x'' * weights_y';
  f ./= sum(f);
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
  nrows_small,ncols_small = size(A);
  nrows_big = 2*nrows_small
  ncols_big = 2*ncols_small
  U = zeros(nrows_big,ncols_big);
  count = 1;
  for i =1:nrows_big
    if i%2!=0
      U[i,1:2:end] = A[count,:];
      count+=1;
    end
  end
  filter = makebinomialfilter(fsize);
  U = imfilter(U,filter,[border="reflect"]);
  U = U.*4.0;

  return U::Array{Float64,2}
end

# Build a gaussian pyramid from an image.
# The output array should contain the pyramid levels in decreasing sizes.
function makegaussianpyramid(im::Array{Float32,2},nlevels::Int,fsize::Array{Int,2},sigma::Float64)
  nrows,ncols = size(im);
  gaussFilter = makegaussianfilter([5 5], 1.5);
  G = Array{Array{Float64, 2},1}(nlevels);
  for i=1:nlevels
    if i==1
      G[i] = Array{Float64,2}(im);
    else
      G[i] = imfilter(G[i-1], gaussFilter, [border="symmetric"]);
      G[i] = downsample2(G[i]);
    end
  end
  return G::Array{Array{Float64,2},1}

end

# Display a given image pyramid (laplacian or gaussian)
function displaypyramid(P::Array{Array{Float64,2},1})
  pyramid = deepcopy(P);

  for i =1:endof(pyramid)
    pyramid[i] = pyramid[i]./sum(pyramid[i]);
  end
  imshow(pyramid[3],"gray",interpolation="none")
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
