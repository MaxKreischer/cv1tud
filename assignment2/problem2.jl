using Images
using PyPlot

# Load images from the yale_faces directory and return a MxN data matrix,
# where M is the number of pixels per face image and N is the number of images.
# Also return the dimensions of a single face image and the number of all face images
function loadfaces()
  n = 0;
  pxPerImg=0;
  facedim = [0 0];
  data=0;
  firsttime = true;
  for (root, dirs, files) in walkdir("./data-julia/yale_faces_png")
    (isempty(dirs))?(break;):();
    for dir in dirs
      path = joinpath(root, dir);
      for (r, d, f) in walkdir(path)
        for file in f
          if firsttime
            firstImgPath = joinpath(r,file);
            firstImg = convert(Array{Float64,2},imread(firstImgPath));
            for (idx, data) in enumerate(size(firstImg))
              facedim[idx] = data;
            end
            pxPerImg = facedim[1]*facedim[2];
            data = reshape(firstImg, pxPerImg);
            firsttime=false;
          end
          imgPath = joinpath(r,file);
          img = convert(Array{Float64,2}, imread(imgPath));
          img_serialized = reshape(img, pxPerImg);
          data = hcat(data, img_serialized);
          n=n+1;
        end
      end
    end
  end
  data = data[:,2:end];
  return data::Array{Float64,2},facedim::Array{Int},n::Int
end

# Apply principal component analysis on the data matrix.
# Return the eigenvectors of covariance matrix of the data, the corresponding eigenvalues,
# the one-dimensional mean data matrix and a cumulated variance vector in increasing order.
function computepca(data::Array{Float64,2})
  (M,N) = size(data);
  mu = sum(data,2)./N;
  zeroMeanData = data - repmat(mu,1,N);
  U,S,V = svd(zeroMeanData);
  lambda = (S.^2)./N;
  cumvar = cumsum(reverse(lambda));
  return U::Array{Float64,2},lambda::Array{Float64,1},mu::Array{Float64,2},cumvar::Array{Float64,1}
end

# Compute required number of components to account for (at least) 80/95 % of the variance
function computencomponents(cumvar::Array{Float64,1})
  thresh_80 = (80/100)*cumvar[end];
  thresh_95 = (95/100)*cumvar[end];
  lam = zeros(size(cumvar)[1]);
  for i=0:(size(cumvar)[1]-2)
    lam[i+1] = cumvar[end-i]-cumvar[end-i-1];
  end
  lam[end] = cumvar[1];
  cumlam = cumsum(lam);
  bThresh80 = [cumlam[i]<=thresh_80 for i=1:size(cumlam)[1]];
  n80 = countnz(bThresh80);
  bThresh95 = [cumlam[i]<=thresh_95 for i=1:size(cumlam)[1]];
  n95 = countnz(bThresh95);

  return n80::Int,n95::Int
end

# Display the mean face and the first 10 Eigenfaces in a single figure
function showfaces(U::Array{Float64,2},mu::Array{Float64,2},facedim::Array{Int})
  dim = (facedim[1],facedim[2]);
  filler = zeros((size(mu)[1],size(mu)[2]*4));
  mFace = hcat(mu,filler);
  #imshow(mFace, "gray", interpolation="none")
  eigenfaces1 = zeros(size(mu)[1],size(mu)[2]*5);
  eigenfaces2=eigenfaces1;
  for i=0:9
    if i==0 || i==5
      eigenfaces1[:,1:dim[2]] = reshape(U[:,1],dim);
      eigenfaces2[:,1:dim[2]] = reshape(U[:,6],dim);
    elseif i<=4
      eigenfaces1[:,(i*dim[2]+1):((i+1)*dim[2])] = reshape(U[:,i+1],dim);
    else
      eigenfaces2[:,((i-5)*dim[2]+1):((i-5+1)*dim[2])] = reshape(U[:,i+1],dim);
    end
  end
  A = vcat(mFace, eigenfaces1,eigenfaces2);
  figure()
  imshow(A,"gray", interpolation="none")
  axis("off")
  return nothing::Void
end

# Fetch a single face with given index out of the data matrix. Returns the actual face image.
function takeface(data::Array{Float64,2},facedim::Array{Int},n::Int)
  face = reshape(data[:,n], (facedim[1],facedim[2]) );
  return face::Array{Float64,2}
end

# Project a given face into the low-dimensional space with a given number of principal
# components and reconstruct it afterwards
function computereconstruction(faceim::Array{Float64,2},U::Array{Float64,2},mu::Array{Float64,2},n::Int)
  dim = (size(faceim)[1],size(faceim)[2]);
  faceim_ser = reshape(faceim,dim[1]*dim[2]);
  mu_ser = reshape(mu,dim[1]*dim[2]);
  face_zMean = faceim_ser - mu_ser;
  B = U[:,1:n];
  face_proj = B'*face_zMean;
  face_rec = mu_ser + B*face_proj;
  recon = reshape(face_rec, dim);
  return recon::Array{Float64,2}
end



# Problem 2: Eigenfaces

function problem2()
  # load data
  data,facedim,N = loadfaces()

  # compute PCA
  U,lambda,mu,cumvar = computepca(data)

  # plot cumulative variance
  figure()
  plot(cumvar)
  grid("on")
  title("Cumulative Variance")
  gcf()

  # compute necessary components for 80% / 95% variance coverage
  n80,n95 = computencomponents(cumvar)

  # plot mean face and first 10 eigenfaces
  showfaces(U,mu,facedim)

  # get a random face
  faceim = takeface(data,facedim,rand(1:N))

  # reconstruct the face with 5,15,50,150 principal components
  f5 = computereconstruction(faceim,U,mu,5)
  f15 = computereconstruction(faceim,U,mu,15)
  f50 = computereconstruction(faceim,U,mu,50)
  f150 = computereconstruction(faceim,U,mu,150)

  # display the reconstructed faces
  figure()
  subplot(221)
  imshow(f5,"gray",interpolation="none")
  axis("off")
  title("5 Principal Components")
  subplot(222)
  imshow(f15,"gray",interpolation="none")
  axis("off")
  title("15 Principal Components")
  subplot(223)
  imshow(f50,"gray",interpolation="none")
  axis("off")
  title("50 Principal Components")
  subplot(224)
  imshow(f150,"gray",interpolation="none")
  axis("off")
  title("150 Principal Components")
  gcf()

  return
end
