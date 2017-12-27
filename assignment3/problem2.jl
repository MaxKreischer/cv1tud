using Images
using PyPlot
using Grid

using JLD

include("Common.jl")


#---------------------------------------------------------
# Loads keypoints from JLD container.
#
# INPUTS:
#   filename     JLD container filename
#
# OUTPUTS:
#   keypoints1   [n x 2] keypoint locations (of left image)
#   keypoints2   [n x 2] keypoint locations (of right image)
#
#---------------------------------------------------------
function loadkeypoints(filename::String)

  keypoints1 = JLD.load(filename, "keypoints1");
  keypoints2 = JLD.load(filename, "keypoints2");

  @assert size(keypoints1,2) == 2
  @assert size(keypoints2,2) == 2
  return keypoints1::Array{Int64,2}, keypoints2::Array{Int64,2}
end


#---------------------------------------------------------
# Compute pairwise Euclidean square distance for all pairs.
#
# INPUTS:
#   features1     [128 x m] descriptors of first image
#   features2     [128 x n] descriptors of second image
#
# OUTPUTS:
#   D             [m x n] distance matrix
#
#---------------------------------------------------------
function euclideansquaredist(features1::Array{Float64,2},features2::Array{Float64,2})
  D = ones(size(features1,2),size(features2,2));
  function eucDist(x,y)
    sum = 0;
    for i=1:128
      sum+= (x[i]-y[i])^2;
    end
    return sum;
  end
  for i=1:size(features1)[2]
    for j=1:size(features2)[2]
      D[i,j] = eucDist(features1[:,i],features2[:,j]);
    end
  end
  @assert size(D) == (size(features1,2),size(features2,2))
  return D::Array{Float64,2}
end


#---------------------------------------------------------
# Find pairs of corresponding interest points given the
# distance matrix.
#
# INPUTS:
#   p1      [m x 2] keypoint coordinates in first image.
#   p2      [n x 2] keypoint coordinates in second image.
#   D       [m x n] distance matrix
#
# OUTPUTS:
#   pairs   [min(N,M) x 4] vector s.t. each row holds
#           the coordinates of an interest point in p1 and p2.
#
#---------------------------------------------------------
function findmatches(p1::Array{Int,2},p2::Array{Int,2},D::Array{Float64,2})
  minDim = minimum(size(D));
  pairs = zeros(Int64,minDim,4);
  matchFirstWithSecond = true;
  (indmin(size(D)) > 1)?(matchFirstWithSecond = false):();

  if(matchFirstWithSecond)
    for i=1:minDim
      pairs[i,1:2] = p1[i,:];
      pairs[i,3:4] = p2[indmin(D[i,:]),:];
    end
  else
    for i=1:minDim
      pairs[i,1:2] = p2[i,:];
      pairs[i,3:4] = p1[indmin(D[i,:]),:];
    end
  end

  @assert size(pairs) == (min(size(p1,1),size(p2,1)),4)
  return pairs::Array{Int,2}
end


#---------------------------------------------------------
# Show given matches on top of the images in a single figure.
# Concatenate the images into a single array.
#
# INPUTS:
#   im1     first grayscale image
#   im2     second grayscale image
#   pairs   [n x 4] vector of coordinates containing the
#           matching pairs.
#
#---------------------------------------------------------
function showmatches(im1::Array{Float64,2},im2::Array{Float64,2},pairs::Array{Int,2})
  figure()
  plotImg = hcat(im1,im2);
  PyPlot.imshow(plotImg,cmap="gray",interpolation="none")
  PyPlot.scatter(pairs[:,1],pairs[:,2])
  PyPlot.scatter(pairs[:,3]+400,pairs[:,4])
  #TODO: Plot lines connecting scattered points
  #PyPlot.plot(pairs[:,1],pairs[:,2],pairs[:,3]+400,pairs[:,4])
  return nothing::Void
end


#---------------------------------------------------------
# Computes the required number of iterations for RANSAC.
#
# INPUTS:
#   p    probability that any given correspondence is valid
#   k    number of samples drawn per iteration
#   z    total probability of success after all iterations
#
# OUTPUTS:
#   n   minimum number of required iterations
#
#---------------------------------------------------------
function computeransaciterations(p::Float64,k::Int,z::Float64)
  #taken from Szeliski book p.319
  n=log(1.0-z)/log(1-(p^k));
  n=Int64.(ceil(n));
  return n::Int
end


#---------------------------------------------------------
# Randomly select k corresponding point pairs.
#
# INPUTS:
#   points1    given points in first image
#   points2    given points in second image
#   k          number of pairs to select
#
# OUTPUTS:
#   sample1    selected [kx2] pair in left image
#   sample2    selected [kx2] pair in right image
#
#---------------------------------------------------------
function picksamples(points1::Array{Int,2},points2::Array{Int,2},k::Int)
  p1Dim = size(points1,1);
  p2Dim = size(points2,1);
  (p1Dim < p2Dim)?(sampleIndices = rand(1:p1Dim,k)):
                                              (sampleIndices = rand(1:p2Dim,k));
  sample1 = zeros(Int64,k,2);
  sample2 = zeros(Int64,k,2);

  for (idx,el) in enumerate(sampleIndices)
    sample1[idx,:] = points1[el,:];
    sample2[idx,:] = points2[el,:];
  end

  @assert size(sample1) == (k,2)
  @assert size(sample2) == (k,2)
  return sample1::Array{Int,2},sample2::Array{Int,2}
end


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
    pts = hcat(points, ones(size(points,1),1))';
    U = zeros(size(points));
    t = zeros(2,1);
    vec = [points[i,:] for i in 1:size(points,1)];
    s = 0.5*maximum(map(norm,vec));
    t[1] = mean(points[:,1]);
    t[2] = mean(points[:,2]);
    T = [1.0/s 0.0 -t[1]/s; 0.0 1.0/s -t[2]/s; 0.0 0.0 1.0];
    Utemp = T*pts;
    U = Utemp[1:2,:]';


  @assert size(U) == size(points)
  @assert size(T) == (3,3)
  return U::Array{Float64,2},T::Array{Float64,2}
end


#---------------------------------------------------------
# Estimates the homography from the given correspondences.
#
# INPUTS:
#   points1    correspondences in left image
#   points2    correspondences in right image
#
# OUTPUTS:
#   H         [3x3] estimated homography
#
#---------------------------------------------------------
function computehomography(points1::Array{Int,2}, points2::Array{Int,2})
  # condition coords.
  #offset = size(points1,1);
  #pts = vcat(points1, points2);
  u1, T1 = condition(Float64.(points1));
  u2, T2 = condition(Float64.(points2));
  # build system of eqs.
  A = zeros(8,9);
  for i=1:4
    x         = u1[i,1];
    x_dashed  = u2[i,1];
    y         = u1[i,2];
    y_dashed  = u2[i,2];
    A[2*i-1,:]    = [0.0 0.0 0.0 x y 1.0 -x*y_dashed -y*y_dashed -y_dashed];
    A[2*i,:]  = [-x -y -1.0 0.0 0.0 0.0 x*x_dashed y*x_dashed x_dashed];
  end
  # solve system
  U,S,V = svd(A, thin=false);
  Hbar = [V[1,9] V[2,9] V[3,9];
          V[4,9] V[5,9] V[6,9];
          V[7,9] V[8,9] V[9,9]];
  H = inv(T2)*Hbar*T1;


  @assert size(H) == (3,3)
  return H::Array{Float64,2}
end


#---------------------------------------------------------
# Computes distances for keypoints after transformation
# with the given homography.
#
# INPUTS:
#   H          [3x3] homography
#   points1    correspondences in left image
#   points2    correspondences in right image
#
# OUTPUTS:
#   d2         distance measure using the given homography
#
#---------------------------------------------------------
function computehomographydistance(H::Array{Float64,2},points1::Array{Int,2},points2::Array{Int,2})
  iterLen = min(size(points1,1),size(points2,1))
  points1_hom = Common.cart2hom(points1');
  points2_hom = Common.cart2hom(points2');
  points1_est = Common.hom2cart(inv(H)*points2_hom);
  points2_est = Common.hom2cart(H*points1_hom);
  d2 = zeros(iterLen,1);

  for i=1:iterLen
      d2[i] = norm(points2_est[:,i] - points2'[:,i])^2 + norm(points1'[:,i] - points1_est[:,i])^2;
  end

  @assert length(d2) == size(points1,1)
  return d2::Array{Float64,2}
end


#---------------------------------------------------------
# Compute the inliers for a given distances and threshold.
#
# INPUTS:
#   distance   homography distances
#   thresh     threshold to decide whether a distance is an inlier
#
# OUTPUTS:
#  n          number of inliers
#  indices    indices (in distance) of inliers
#
#---------------------------------------------------------
function findinliers(distance::Array{Float64,2},thresh::Float64)
  n = 0;
  indices = Array{Int64,1}();
  for (idx,el) in enumerate(distance)
    if(el < thresh)
      n+=1;
      push!(indices, idx);
    end
  end

  return n::Int,indices::Array{Int,1}
end


#---------------------------------------------------------
# RANSAC algorithm.
#
# INPUTS:
#   pairs     potential matches between interest points.
#   thresh    threshold to decide whether a homography distance is an inlier
#   n         maximum number of RANSAC iterations
#
# OUTPUTS:
#   bestinliers   [n x 1 ] indices of best inliers observed during RANSAC
#
#   bestpairs     [4x4] set of best pairs observed during RANSAC
#                 i.e. 4 x [x1 y1 x2 y2]
#
#   bestH         [3x3] best homography observed during RANSAC
#
#---------------------------------------------------------
function ransac(pairs::Array{Int,2},thresh::Float64,n::Int)
  img1_pts = pairs[:,1:2];
  img2_pts = pairs[:,3:4];
  nCurrBestInliers = 0;
  bestinliers = Array{Int,1}();
  bestpairs = Array{Int,2}();
  bestH = Array{Float64,2}();
  for idx = 1:n
    sample1,sample2 = picksamples(img1_pts,img2_pts,4);
    try
      H = computehomography(sample1,sample2);
      d2 = computehomographydistance(H, img1_pts, img2_pts);
      nInliers, currIndices = findinliers(d2, thresh);
      if (nCurrBestInliers < nInliers)
        bestinliers = currIndices;
        bestpairs = hcat(sample1,sample2);
        bestH = H;
      end
    catch linAlgError
      if( isa(linAlgError, Base.LinAlg.SingularException) )
        println("Singular Matrix Error")
      else
        println("Other error")
      end
    end

  end

  @assert size(bestinliers,2) == 1
  @assert size(bestpairs) == (4,4)
  @assert size(bestH) == (3,3)
  return bestinliers::Array{Int,1},bestpairs::Array{Int,2},bestH::Array{Float64,2}
end


#---------------------------------------------------------
# Recompute the homography based on all inliers
#
# INPUTS:
#   pairs     pairs of keypoints
#   inliers   inlier indices.
#
# OUTPUTS:
#   H         refitted homography using the inliers
#
#---------------------------------------------------------
function refithomography(pairs::Array{Int64,2}, inliers::Array{Int64,1})
  bestPairs = zeros(Int64, size(inliers,1),4);
  [bestPairs[i,:] = pairs[inliers[i],:] for i=1:size(inliers,1)];
  H = computehomography(bestPairs[:,1:2],bestPairs[:,3:4]);

  @assert size(H) == (3,3)
  return H::Array{Float64,2}
end


#---------------------------------------------------------
# Show panorama stitch of both images using the given homography.
#
# INPUTS:
#   im1     first grayscale image
#   im2     second grayscale image
#   H       [3x3] estimated homography between im1 and im2
#
#---------------------------------------------------------
function showstitch(im1::Array{Float64,2},im2::Array{Float64,2},H::Array{Float64,2})
  #transform img2plane to img1plane
  xrange = 1:size(im2,2);
  yrange = 1:size(im2,1);
  nPoints = size(im2,1)*size(im2,2);

  coords = zeros(nPoints,2);
  coords = [[x y]  for x in xrange for y in yrange];
  coords = Array{Float64,2}.(coords);
  coords2 = zeros(nPoints,3);
  for (idx,el) in enumerate(coords )
    coords2[idx,:] = hcat(coords[idx], 1.0);
  end
  coordsT = inv(H)*coords2';

  for col in 1:size(coordsT,2)
    coordsT[:,col] = coordsT[:,col]./coordsT[3,col];
  end

  rows = 1:size(im2,1);
  cols = 1:size(im2,2);
  interGrid = CoordInterpGrid((rows,cols), im2, BCnan, InterpLinear)
  newimg =  zeros(size(im2))
  currRow = 1
  currCol = 1
  for i=1:size(coordsT,2)
    if i%size(im2,1)==0
      currRow = 1;
      currCol +=1;
    end
    (currCol > 400)?(break):();
    newimg[currRow,currCol] = interGrid[ coordsT[2,i],coordsT[1,i]-coordsT[1] ]
    currRow+=1;
  end

  return coordsT,newimg
  #return nothing::Void
end


#---------------------------------------------------------
# Problem 2: Image Stitching
#---------------------------------------------------------
function problem2()
  # SIFT Parameters
  sigma = 1.4             # standard deviation for presmoothing derivatives

  # RANSAC Parameters
  ransac_threshold = 50.0 # inlier threshold
  p = 0.5                 # probability that any given correspondence is valid
  k = 4                   # number of samples drawn per iteration
  z = 0.99                # total probability of success after all iterations

  # load images
  im1 = PyPlot.imread("a3p2a.png")
  im2 = PyPlot.imread("a3p2b.png")

  # Convert to double precision
  im1 = Float64.(im1)
  im2 = Float64.(im2)

  # load keypoints
  keypoints1, keypoints2 = loadkeypoints("keypoints.jld")

  # extract SIFT features for the keypoints
  features1 = Common.sift(keypoints1,im1,sigma)
  features2 = Common.sift(keypoints2,im2,sigma)

  # compute chi-square distance  matrix
  D = euclideansquaredist(features1,features2)

  # find matching pairs
  pairs = findmatches(keypoints1,keypoints2,D)

  # show matches
  showmatches(im1,im2,pairs)
  title("Putative Matching Pairs")

  # compute number of iterations for the RANSAC algorithm
  niterations = computeransaciterations(p,k,z)

  # apply RANSAC
  bestinliers,bestpairs,bestH = ransac(pairs,ransac_threshold,niterations)

  # show best matches
  showmatches(im1,im2,bestpairs)
  title("Best 4 Matches")

  # show all inliers
  showmatches(im1,im2,pairs[bestinliers,:])
  title("All Inliers")

  # stitch images and show the result
  showstitch(im1,im2,bestH)

  # recompute homography with all inliers
  H = refithomography(pairs,bestinliers)
  showstitch(im1,im2,H)

  return nothing
end

function showcrap(im1,im2)
  figure()
  subplot(211)
  imshow(im1,cmap="gray",interpolation="none")
  axis("off")
  subplot(212)
  imshow(im2,cmap="gray",interpolation="none")
  axis("off")
end


function testFun(coords,img)
  xrange = 1:size(img,1);
  yrange = 1:size(img,2);
  interGrid = CoordInterpGrid((xrange,yrange), img, BCnan, InterpLinear)
  newimg =  zeros(size(img))
  currRow = 1
  currCol = 1
  for i=1:size(coords,2)
    if i%size(img,1)==0
      currRow = 1;
      currCol +=1;
    end
    (currCol > 400)?(break):()
    newimg[currRow,currCol] = interGrid[coords[2,i],coords[1,i]-184.494]
    currRow+=1;
  end
  return newimg
end
function showCrappy(img1,img2New)
  pano = zeros(size(img1,1),700);
  for i=1:300
    pano[:,i] = img1[:,i]
  end
  currCol = 301
  for i = 100:size(img2New,2)
    #(currCol >700)?(break):();
    pano[:,currCol] = img2New[:,i];
    currCol+=1;
  end
  imshow(pano, cmap="gray", interpolation="none")
  return nothing
end
