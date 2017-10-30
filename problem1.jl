using PyPlot
using JLD

# load and return the given image
function loadimage()
  img = PyPlot.imread("a1p1.png");
  return img::Array{Float32,3}
end


# save the image as a .jld file
function savefile(img::Array{Float32,3})
  JLD.save("./img.jld", "img", img::Array{Float32,3})
end


# load and return the .jld file
function loadfile()
  img = JLD.load("./img.jld", "img")
  return img::Array{Float32,3}
end


# create and return a vertically mirrored image
function mirrorvertical(img::Array{Float32,3})
  nrows, ncols, _ = size(img)
  mirrored = Array{Float32}(size(img))
  for rows=nrows:-1:1
    mirrored[(nrows-rows+1),:,:] = img[rows,:,:]
  end
  return mirrored::Array{Float32,3}
end


# display the normal and the mirrored image in one plot
function showimages(img1::Array{Float32,3}, img2::Array{Float32,3})
  # Sources used for plot creation:
  #https://stackoverflow.com/questions/35692507/plot-several-image-files-in-matplotlib-subplots
  #And:
  #https://github.com/gizmaa/Julia_Examples/blob/master/pyplot_subplot.jl
  fig = figure("pyplot_subplot_column")
  subplot(211)
  PyPlot.imshow(img1)
  title("Non-inverted")
  PyPlot.axis("off")
  subplot(212)
  PyPlot.imshow(img2)
  title("Inverted")
  PyPlot.axis("off")
  fig[:canvas][:draw]() # Update the figure
end


#= Problem 1
Load and Display =#

function problem1()

  img1 = loadimage()

  savefile(img1)

  img2 = loadfile()

  img2 = mirrorvertical(img2)

  showimages(img1, img2)



end
