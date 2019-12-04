# import the necessary packages
import imutils
import cv2
import os
import sys

class ImageCropper:
	# init
	def __init__(self, input_folder, output_folder, dirs):
		self.input_folder = input_folder
		self.output_folder = output_folder
		self.dirs = dirs
	
	
	# creates the required directories
	def createOutputDirectory(self):
	
		try:
			os.mkdir(self.output_folder)
		except:
			print(self.output_folder+" already exists")

		for dir in self.dirs:
			try:
				os.mkdir(os.path.join(self.output_folder, dir))
			except:
				print(os.path.join(self.output_folder, dir)+" already exists")


	# crops every image in the listed subfolders
	def cropImages(self):
	
		for dir in self.dirs:
			for filename in os.listdir(os.path.join(self.input_folder, dir)):
				print(filename)
				image = cv2.imread(os.path.join(self.input_folder, dir, filename))

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

				# find contours in thresholded image, then grab the largest one
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
				
				# write the image to disk
				cv2.imwrite(os.path.join(self.output_folder, dir, filename), img_crop)


def main():

	# check the number of arguments
	if len(sys.argv) != 3:
		print("Error expected 2 arguments but got "+str(len(sys.argv)-1)+" instead!")
		print("Expected arguments: dataset\ output_folder\\")
		exit(0)
		
	INPUT_DIRECTORY = sys.argv[1]
	OUTPUT_DIRECTORY = sys.argv[2]
	DIRS = ["yes", "no"]

	# creates the cropper object
	imgc = ImageCropper(INPUT_DIRECTORY, OUTPUT_DIRECTORY, DIRS)
	
	# creates the output folders
	imgc.createOutputDirectory()
	
	# does the job
	imgc.cropImages()
	
	
if __name__ == "__main__":
	main()