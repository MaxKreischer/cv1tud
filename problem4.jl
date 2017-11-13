using Images
using PyPlot

#Sources: http://www.cs.cornell.edu/courses/cs6670/2011sp/lectures/lec02_filter.pdf
# Create 3x3 derivative filters in x and y direction
function createfilters()
  fx = [-1 0 1; -2 0 2; -1 0 1];
  fx =  fx/8.0;
  fy = [1 2 1; 0 0 0; -1 -2 -1];
  fy = fy/8.0;
  return fx::Array{Float64,2},fy::Array{Float64,2}
end


# Apply derivate filters to an image and return the derivative images
function filterimage(I::Array{Float32,2},fx::Array{Float64,2},fy::Array{Float64,2})
  Ix = imfilter(I, fx, [border="replicate"]);
  Iy = imfilter(I, fy, [border="replicate"]);
  return Ix::Array{Float64,2},Iy::Array{Float64,2}
end


# Apply thresholding on the gradient magnitudes to detect edges
function detectedges(Ix::Array{Float64,2},Iy::Array{Float64,2}, thr::Float64)
  nrows,ncols = size(Ix);
  magnitude = sqrt(Ix.^2 + Iy.^2);
  edges = zeros(nrows,ncols);
  for col=1:ncols
    for row=1:nrows
      (magnitude[row,col] >= thr)?(edges[row,col] = 1.0):(edges[row,col] = 0.0);
    end
  end
  return edges::Array{Float64,2}
end


# Apply non-maximum-suppression
function nonmaxsupp(edges::Array{Float64,2},Ix::Array{Float64,2},Iy::Array{Float64,2})
  #Source: https://en.wikipedia.org/wiki/Canny_edge_detector#Non-maximum_suppression
  nrows,ncols = size(Ix);
  magnitude = sqrt(Ix.^2 + Iy.^2);
  compareVal1 = 0.0;
  compareVal2 = 0.0;
  edgeCopy = zeros(nrows,ncols);
  for col=2:ncols-1
      for row=2:nrows-1
        pixel = magnitude[row,col];
        gradDirection = atan2(Iy[row,col],Ix[row,col])*(180/pi);
        (gradDirection < 0)?(gradDirection = 360+gradDirection):(gradDirection=gradDirection);
        if     (0<= gradDirection <= 45) | (180 <= gradDirection <= 225)
          # E,NE & W,SW
          compareVal1 = 0.5*(magnitude[row, col+1] + magnitude[row-1, col+1]);
          compareVal2 = 0.5*(magnitude[row, col-1] + magnitude[row+1, col-1]);
        elseif (45<= gradDirection <= 90) | (225 <= gradDirection <= 270)
          # NE,N & SW,S
          compareVal1 = 0.5*(magnitude[row-1, col+1] + magnitude[row-1, col]);
          compareVal2 = 0.5*(magnitude[row+1, col-1] + magnitude[row+1, col]);
        elseif (90<= gradDirection <= 135) | (270 <= gradDirection <= 315)
          # N,NW & S,SE
          compareVal1 = 0.5*(magnitude[row-1, col] + magnitude[row-1, col-1]);
          compareVal2 = 0.5*(magnitude[row+1, col] + magnitude[row+1, col+1]);
        elseif (135<= gradDirection <= 180) | (315 <= gradDirection <= 360)
          # NW,W & SE,E
          compareVal1 = 0.5*(magnitude[row-1, col-1] + magnitude[row, col-1]);
          compareVal2 = 0.5*(magnitude[row+1, col+1] + magnitude[row, col+1]);
        end
        if (pixel > compareVal1) & (pixel > compareVal2) & (edges[row,col]>0.0)
          edgeCopy[row,col] = 1.0
        end

      end
  end
  return edgeCopy::Array{Float64,2}
end


#= Problem 4
Image Filtering and Edge Detection =#

function problem4()
  # load image
  img = PyPlot.imread("a1p4.png")

  # create filters
  fx, fy = createfilters()

  # filter image
  imgx, imgy = filterimage(img, fx, fy)

  # show filter results
  figure()
  subplot(121)
  imshow(imgx, "gray", interpolation="none")
  title("x derivative")
  axis("off")
  subplot(122)
  imshow(imgy, "gray", interpolation="none")
  title("y derivative")
  axis("off")
  gcf()

  # show gradient magnitude
  figure()
  imshow(sqrt(imgx.^2 + imgy.^2),"gray", interpolation="none")
  axis("off")
  title("Derivative magnitude")
  gcf()

  # threshold derivative
  # Threshold Value:
  # Maximum magnitude is at ~0.421, this evaluates to an edge
  # of 1 pixel, thus a hand tuned smaller value is chosen;
  # About a quarter of the maximum seems to yield acceptable results.
  threshold = 420/(4*1000);

  edges = detectedges(imgx,imgy,threshold)

  figure()
  imshow(edges.>0, "gray", interpolation="none")
  axis("off")
  title("Binary edges")
  gcf()

  # non maximum suppression
  edges2 = nonmaxsupp(edges,imgx,imgy)
  figure()
  imshow(edges2,"gray", interpolation="none")
  axis("off")
  title("Non-maximum suppression")
  gcf()

  return
end
