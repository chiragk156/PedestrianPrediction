import cv2
import numpy as np
import sys
import os
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split

def foreground(imgo):
	height, width = imgo.shape[:2]

	#Create a mask holder
	mask = np.zeros(imgo.shape[:2],np.uint8)

	#Grab Cut the object
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)

	#Hard Coding the Rect… The object must lie within this rect.
	rect = (10,10,width-30,height-30)
	cv2.grabCut(imgo,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
	mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	img1 = imgo*mask[:,:,np.newaxis]

	# #Get the background
	# background = imgo - img1

	# #Change all pixels in the background that are not black to white
	# background[np.where((background > [0,0,0]).all(axis = 2))] = [0,0,0]

	min_x = height
	min_y = width
	max_x = 0
	max_y = 0

	for i in range(height):
		for j in range(width):
			if img1[i][j][0] != 0 or img1[i][j][1] != 0 or img1[i][j][2] != 0:
				if i < min_x:
					min_x = i
				if i > max_x:
					max_x = i
				if j < min_y:
					min_y = j
				if j > max_y:
					max_y = j

	#Add the background and the image
	# final = background + img1

	#To be done – Smoothening the edges….

	return imgo[min_x:max_x,min_y:max_y,:]

def get_Hog_Descriptor(img):
	img_64_128 = cv2.resize(img,(64,128))
	hog = cv2.HOGDescriptor()
	return hog.compute(img_64_128)

def get_sift_descriptor(img):
	pass

mydir = sys.argv[1]

X = []
Y = []

directions = [0,45,90,135,180,225,270,315]
for x in directions:
	direction_directory = os.path.join(mydir,str(x))

	for filename in os.listdir(direction_directory):
		filepath = os.path.join(direction_directory, filename)
		img = cv2.imread(filepath)
		temp = []
		for i in get_Hog_Descriptor(img):
			temp.append(i[0])

		# Gaussian Pyramid1
		img1 = cv2.pyrDown(img)

		for i in get_Hog_Descriptor(img1):
			temp.append(i[0])

		# Gaussian Pyramid2
		img2 = cv2.pyrDown(img1)

		for i in get_Hog_Descriptor(img2):
			temp.append(i[0])

		# # Gaussian Pyramid3
		# img3 = cv2.pyrDown(img2)

		# for i in get_Hog_Descriptor(img3):
		# 	temp.append(i[0])

		X.append(temp)
		Y.append(x)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

clf = svm.SVC(kernel='linear')
# clf = svm.SVC(kernel='poly', degree=3) 
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))



# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,2,3,0.04)

# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)

# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]

# cv2.imshow('dst',img)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
# cv2.imshow('image',foreground(img))

# k = cv2.waitKey(0)

# if k==27:
# 	cv2.destroyAllWindows()