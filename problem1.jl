using PyPlot
using JLD


# load and return the given image
function loadimage()

  return img::Array{Float32,3}
end


# save the image as a .jld file
function savefile(img::Array{Float32,3})

end


# load and return the .jld file
function loadfile()

  return img::Array{Float32,3}
end


# create and return a vertically mirrored image
function mirrorvertical(img::Array{Float32,3})

  return mirrored::Array{Float32,3}
end


# display the normal and the mirrored image in one plot
function showimages(img1::Array{Float32,3}, img2::Array{Float32,3})


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
