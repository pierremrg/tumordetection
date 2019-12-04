# USAGE
# python extreme_points.py

# import the necessary packages
import imutils
import cv2
import os
import sys

if len(sys.argv) != 3:
	print("Error expected 2 arguments but got "+str(len(sys.argv)-1)+" instead!")
	print("Expected arguments: input_folder(dataset) output_folder(cropped images)")
	exit(0)
	
input_folder = sys.argv[1]+"\\"
output_folder = sys.argv[2]+"\\"

for filename in os.listdir(input_folder):
	print(filename)
	image = cv2.imread(input_folder+filename)

	# the file couldn't be read as an image or doesn't exist
	if (image is None):
		print("Error opening the file, skipping "+filename)
		continue
	
	# image is converted to grayscale and slightly blurred
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)

	# threshold the image, then perform a series of erosions +
	# dilations to remove any small regions of noise
	thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)

	# find contours in thresholded image, then grab the largest
	# one
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)

	# determine the most extreme points along the contour
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])

	# determine the vertices of the corresponding box
	topLeft = (extLeft[0], extTop[1])
	topRight = (extRight[0], extTop[1])
	botLeft = (extLeft[0], extBot[1])
	botRight = (extRight[0], extBot[1])

	# crop the image to the size of the box
	img_crop = image[ topLeft[1]:botLeft[1], topLeft[0]:topRight[0] ]
	#cv2.imshow("Image", image)
	#cv2.imshow("Image crop", img_crop)
	#cv2.waitKey(0)
	
	# write the image to disk
	cv2.imwrite(output_folder+filename, img_crop)
