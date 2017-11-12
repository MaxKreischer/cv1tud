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

  return edges::Array{Float64,2}
end


# Apply non-maximum-suppression
function nonmaxsupp(edges::Array{Float64,2},Ix::Array{Float64,2},Iy::Array{Float64,2})

  return edges::Array{Float64,2}
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
#  gcf()
"""
  # show gradient magnitude
  figure()
  imshow(sqrt(imgx.^2 + imgy.^2),"gray", interpolation="none")
  axis("off")
  title("Derivative magnitude")
  gcf()
  # threshold derivative
  threshold = 42
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
"""
  return
end
