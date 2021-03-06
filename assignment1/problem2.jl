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

#function getSizeFromArray(someImage::Array)

function zeroBounds(imgReceived::Array{Float64,2})
    #source: script l3. Implementation of solving boundaryissues by adding zeros
    tempRows, tempCols = size(imgReceived)
    zeroFilled = zeros(Float64,tempRows+2,tempCols+2)
    #copy Values to inner Area of ZeroBoundsArry
    for rows = 1 : tempRows
        for cols = 1 : tempCols
            zeroFilled[rows+1,cols+1] = imgReceived[rows,cols]
        end
    end
    return zeroFilled
end

function bilinearInterpolation(zeroBoundedImage::Array{Float64,2})

    nrows, ncols = size(zeroBoundedImage)
    interpolatedImage = zeros(nrows,ncols)
    zeroCounter = 0
    temp = 0.0
    temp2 = 0.0
    for rows = 2 : nrows-1
        for cols = 2 : ncols-1
            if (zeroBoundedImage[rows,cols]==0)
              zeroCounter = 0
              temp = zeroBoundedImage[rows-1,cols]+zeroBoundedImage[rows,cols-1]+zeroBoundedImage[rows,cols+1]+zeroBoundedImage[rows+1,cols]
              temp2 = zeroBoundedImage[rows-1,cols-1]+zeroBoundedImage[rows-1,cols+1]+zeroBoundedImage[rows+1,cols-1]+zeroBoundedImage[rows+1,cols+1]

              if (temp!=0)#nondiagonal pattern
                  #check what amount of values is used for interpolation
                  if (zeroBoundedImage[rows-1, cols] == 0) zeroCounter = zeroCounter+1 end
                  if (zeroBoundedImage[rows+1, cols] == 0) zeroCounter = zeroCounter+1 end
                  if (zeroBoundedImage[rows, cols-1] == 0) zeroCounter = zeroCounter+1 end
                  if (zeroBoundedImage[rows, cols+1] == 0) zeroCounter = zeroCounter+1 end

                  if (zeroCounter == 0) temp = temp/4 end
                  if (zeroCounter == 1) temp = temp/3 end #somekind of errormessage
                  if (zeroCounter == 2) temp = temp/2  end
                  if (zeroCounter == 3) temp = temp  end
                  if (zeroCounter == 4) temp = temp  end #somekind of errormessage

                  interpolatedImage[rows, cols] = temp
              else
                  #check what amount of values is used for interpolation
                  if (zeroBoundedImage[rows-1,cols-1] == 0) zeroCounter = zeroCounter+1 end
                  if (zeroBoundedImage[rows-1,cols+1] == 0) zeroCounter = zeroCounter+1 end
                  if (zeroBoundedImage[rows+1,cols-1] == 0) zeroCounter = zeroCounter+1 end
                  if (zeroBoundedImage[rows+1,cols+1] == 0) zeroCounter = zeroCounter+1 end

                  if (zeroCounter == 0) temp2 = temp2/4 end
                  if (zeroCounter == 1) temp2 = temp2/3 end #somekind of errormessage
                  if (zeroCounter == 2) temp2 = temp2/2  end
                  if (zeroCounter == 3) temp2 = temp2  end
                  if (zeroCounter == 4) temp2 = temp2  end #somekind of errormessage

                  interpolatedImage[rows, cols] = temp2
              end #end of checking which was used

              else  interpolatedImage[rows, cols] = zeroBoundedImage[rows, cols]
            end
        end
    end

      return interpolatedImage
end

#inverting method of zeroBounds. now the zerobounds are getting removed
function removeZeroBoundary(zeroBoundedImage::Array{Float64,2})
  recRows, recCols = size(zeroBoundedImage::Array{Float64,2})
  reducedMatrix = zeros(Float64, recRows-2, recCols-2)

  #copy Values to inner Area of ZeroBoundsArry
  for rows = 1 : recRows-2
      for cols = 1 : recCols-2
          reducedMatrix[rows,cols] = zeroBoundedImage[rows+1,cols+1]
      end
  end
  return reducedMatrix
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

  #kerG  =      castMat2Float(0.25*[0 1 0; 1 4 1; 0 1 0])
  #kerC  =      castMat2Float(0.25*[1 2 1; 2 4 2; 1 2 1])

  #Solving bounderyissues using zerofilling
  #received matrices seem to be right
  zeroR = zeroBounds(r)
  zeroG = zeroBounds(g)
  zeroB = zeroBounds(b)

  #using interpolation method

  interpolatedR = bilinearInterpolation(zeroR)
  interpolatedG = bilinearInterpolation(zeroG)
  interpolatedB = bilinearInterpolation(zeroB)

  #removing the bounds to old arraysize 480x320

  originDimensionsR = removeZeroBoundary(interpolatedR)
  originDimensionsG = removeZeroBoundary(interpolatedG)
  originDimensionsB = removeZeroBoundary(interpolatedB)


  #imRed   =   r + Images.imfilter(r, kerNN)
  #imGreen =   g+ Images.imfilter(g, kerNN)
  #imBlue  =   b + Images.imfilter(b, kerNN)
  #rFilt = zeros(nrows,ncols)
  #gFilt = zeros(nrows,ncols)
  #bFilt = zeros(nrows,ncols)



  #image = makeimage(imRed, imGreen, imBlue)
  image = makeimage(originDimensionsR, originDimensionsG, originDimensionsB)
  return image
  #image::Array{Float64,3}
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

  #PyPlot.imshow(b, cmap="gray")
  img1 = makeimage(r,g,b)
  # interpolate Bayer
  img2 = debayer(r,g,b)
  #zeroed, interpolatedd = debayer(r,g,b)

  #PyPlot.imshow(img2, interpolation="none")
  # display images
  displayimages(img1,img2)
  #BoundsTester = zeroBounds(g)
  return
end
