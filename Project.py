#Importing required libraries
import numpy as np
import cv2
import os

#Storing the path to the haar cascade file on disk as a variable
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, "C:\python39\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
#Loading the required face classifier from OpenCV
detector = cv2.CascadeClassifier(haar_model)

#Holds current ID of person in Image
currentID = 0

#Dictionary to hold users names and number associated with each
labelIDs = {}

#Creating empty list for image paths
images= []

#List to hold image labels
imageLabels = []

#List holding the face region of the images
imageFaces = []

#List holding the face regions all resized to the same size
resizedFaces = []

#List holding flattened face images
vectorisedFaces = []

#List holding mean reduced face images
normalisedImages = []


#Setting base directory of subject images
basePath = "Training data/"

#Looping through each folder/files in the basepath 
for root, dirs, files in os.walk(basePath):
    for file in files:
        #Only continuing for files ending with 'png' or 'jpg' extension
        if file.endswith('png') or file.endswith('jpg'):
            #Saving the image path to a variable and saving the basename of each path
            #(the images) as labels
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path))

            if not label in labelIDs:
                
                labelIDs[label] = currentID
                currentID += 1
            ID = labelIDs[label]
            #print(labelIDs)

            #Reading each image into the 'image' variable
            image = cv2.imread(path)
            #Turning each image into a numpy array of type 'uint8'
            imageArray = np.array(image, 'uint8')

            #Applying the face classifier to the training dataset
            faces = detector.detectMultiScale(
		imageArray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))

            #Saving the ROI in each image to a variable and appending to a list,
            #each image has its label appended to a seperate list
            for (x,y,w,h) in faces:
                roi = imageArray[y:y+h, x:x+w]
                imageFaces.append(roi)
                imageLabels.append(ID)
                #print(ID)

            #Applying a rectangle around the ROI on each image as a
            #visual aid to where one has been found
            for (x, y, w, h) in faces:
                    cv2.rectangle(imageArray, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    #cv2.imshow('Images', imageArray)
                    #cv2.waitKey(0)
    
#This function resizes the ROI of the images to the same size 
def image_resize(face, width = None, height = None, inter = cv2.INTER_AREA):
    # Initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = face.shape[:2]

    # If both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return face

    # Check to see if the width is None
    if width is None:
        # Calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # Otherwise, the height is None
    else:
        # Calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Resize the image
    resized = cv2.resize(face, dim, interpolation = inter)

    # Return the resized image
    return resized


#Looping through each image and resizing to the same size
for i in range(len(imageFaces)):
    facesRes = image_resize(imageFaces[i], height = 270)
    resizedFaces.append(facesRes)
##    print(facesRes.shape)
##    cv2.imshow('Resized ROI', facesRes)
##    cv2.waitKey(0)

    

#Converting each image to grayscale
for i in range(len(resizedFaces)):
    grayFace = cv2.cvtColor(resizedFaces[i], cv2.COLOR_BGR2GRAY)
    #print(grayFace.shape)
##    cv2.imshow('faces', grayFace)
##    cv2.waitKey(0)
    #Flattening each image into a column vector then appending to a list
    flattenedFace = grayFace.flatten()
    vectorisedFaces.append(flattenedFace)
    #print(flattenedFace.shape)

#Changing list of flattened images into a numpy array
vectorisedFaces = np.array(vectorisedFaces)

#Calculating the mean of all the flattened images
meanImage = np.mean(vectorisedFaces, axis=0)
###Setting the datatype of the mean image to 'uint8' as this
###is the datatype used to display gray images with openCV
meanImage = meanImage.astype('uint8')
###Reshaping the image to the original images layout
meanImage = meanImage.reshape(grayFace.shape)
#######Uncomment below to output the mean image#######

####print(meanImage.shape)
####print(meanImage)
#####Outputting average face image
##cv2.imshow('Mean image', meanImage)
##cv2.waitKey(0)

#Reshaping each flattened image to original image shape then
#subtracting the mean from each image to normalise the dataset
for i in range(len(vectorisedFaces)):
    imReShape = np.reshape(vectorisedFaces[i], (grayFace.shape))
##    cv2.imshow('d', imReShape)
##    cv2.waitKey(1500)
    normalisedImage = np.subtract(imReShape, meanImage)
##    print(normalisedImage.shape)
##    print(normalisedImage)
##    cv2.imshow('Normalised images', normalisedImage)
##    cv2.waitKey(0)
    #Reshaping normalised images to column vectors then
    #appending each to a list
    normalisedImage = normalisedImage.reshape(flattenedFace.shape)
    normalisedImages.append(normalisedImage)

#Converting list of normalised images to a numpy array
normalisedImages = np.array(normalisedImages)

#Calculating the covariance matrix of the normalised images 
covarianceMatrix = np.cov(normalisedImages)
#print(covarianceMatrix.shape)
#print(covarianceMatrix)

#Extracting the eigenvectors from the covariance matrix with their
#corresponding eigenvalues
eigenvals, eigenvecs = np.linalg.eig(covarianceMatrix)

# Sort the eigen values/vectors in descending order:
idx = np.argsort(-eigenvals)
eigenvals = eigenvals[idx]
eigenvecs = eigenvecs[:, idx]

#Calculating eigenvectors of larger covariance matrix (eigenfaces)
p = np.dot(eigenvecs,normalisedImages)

#######Uncomment below to view all eigenfaces#######
##for i in range(len(eigenvecs)):
##    eigenface = p[i].reshape(grayFace.shape)
##    eigenface = eigenface.astype('uint8')
##
##
##    cv2.imshow('Eigenfaces', eigenface)
##    cv2.waitKey(0)

#Calculating the amount the variance is explained by each component
variancePercentage = []
for i in eigenvals:
    variancePercentage.append((i/sum(eigenvals))*100)
#print(variancePercentage)

#Displaying the cumulative percentage explained by each feature
#to determine the number of features that need to be kept
cumulativeVariance = np.cumsum(variancePercentage)
#print(cumulativeVariance)

###Setting number of principal components to use
###based on the above cumulative variance results
num_components = 25

#Applying the 'num_components' value to
#get only the most useful eigenfaces
p = p[0:num_components, :].copy()

#Uncomment beow to view the top eigenfaces
##for i in range(num_components):
##    eigenface = p[i].reshape(grayFace.shape)
##    eigenface = eigenface.astype('uint8')
##    print(eigenface.shape)
##    cv2.imshow('Eigenfaces', eigenface)
##    cv2.waitKey(0)
 

#Calculating weights vector for each normalised image in  the dataset
weights = np.array([np.dot(p, i) for i in normalisedImages])

#####FACE RECOGNITION SECTION#######

#Setting camera capture from webcam
cap = cv2.VideoCapture(0)

#Loop continues until capture is set to stop
while (cap.isOpened()):
    #Reads each frame from camera feed
    ret, frame = cap.read()

    #Captures ROI from user's face in video feed
    testFace = detector.detectMultiScale(
		frame, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))

    #Extracting ROI from image and resizing to match size of training images
    for (x,y,w,h) in testFace:
         testRoi = frame[y:y+h, x:x+w]
         testFaceRes = image_resize(testRoi, height = 270)

         #Applying a rectangle around the ROI in each frame           
         for (x, y, w, h) in testFace:
             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

         #Converting frame to grayscale
         grayFrame = cv2.cvtColor(testFaceRes, cv2.COLOR_BGR2GRAY)
        
         #Normalising frame by subtracting mean image
         normalisedFrame = grayFrame - meanImage
         #Reshaping normalised frame into a column vector
         normalisedFrame = normalisedFrame.reshape(flattenedFace.shape)

         #Calculating weights for the normalised frame
         unknownWeights = np.dot(p, normalisedFrame)
         
         #Calculating the euclidean distance between the frames weights and each of the
         #weights of the test images
         dist = np.sqrt(np.sum(np.asarray(weights-unknownWeights.T)**2, axis = 1))

         #Setting a threshold on whether a face is known or unknown
         threshold = 3886447271.603497
         
         #Checking if the largest distance between weights values
         #are above the set threshold
         if max(dist) < threshold:
             #Finding which image is the closest in weights values
             minDistIndex = np.argmin(dist)
             #Finding the exact image associated with the closest weights values
             match = imageFaces[minDistIndex]
             #Finding the label for the matched image
             nameNo = imageLabels[minDistIndex]
             name = list(labelIDs.keys())[list(labelIDs.values()).index(nameNo)]
             #If the distance between weights is above the threshold
             #it is given a 'name' of one higher than the number of known
             #faces to identify it as an unknown face
         else:
             name = max(imageLabels) + 1
                 
         
         #Setting font for name on frame
         font = cv2.FONT_HERSHEY_SIMPLEX
         #Setting colour for name on frame
         colour = (255, 0, 0)

         #Applying persons name to the frame if known and applying "Unknown face"
         #if unknown
         if name == max(imageLabels) + 1:
             cv2.putText(frame, "Unknown face", (x,y), font, 1, colour, 2, cv2.LINE_AA)
         else:
             cv2.putText(frame, name, (x,y), font, 1, colour, 2, cv2.LINE_AA)

    #Outputting frame to user in feed
    cv2.imshow("Frame", frame)
    #If user presses 'q' key the video feed ends
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
    
#End video feed and destroy any open windows
cap.release()    
cv2.destroyAllWindows()










