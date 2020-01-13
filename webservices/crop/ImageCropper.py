# import the necessary packages
import imutils
import cv2
import os
import sys
import logging
import numpy as np

from hdfs import InsecureClient

formatter = logging.Formatter(fmt = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

class ImageCropper:
	# init
	def __init__(self, input_folder, image_path, output_folder):
		logging.info('ImageCropper.init')

		self.input_folder = input_folder
		self.image_path = image_path
		self.output_folder = output_folder
		self.hdfs_client = InsecureClient('http://192.168.1.4:9870', user='hadoop')


	# crops every image in the listed subfolders
	def cropImages(self):
		logging.info('ImageCropper.cropImages')

		for filename in self.image_path:
			with self.hdfs_client.read('/' + self.input_folder + filename) as reader:
				image = np.asarray(bytearray(reader.read()), dtype='uint8')
				image = cv2.imdecode(image, cv2.IMREAD_COLOR)

			# the file couldn't be read as an image or doesn't exist
			if (image is None):
				loggin.info("Error opening the file, skipping "+filename)
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

			cv2.imwrite(filename.split('/')[-1], img_crop)
			self.hdfs_client.upload('/' + self.output_folder + filename, filename.split('/')[-1])
			os.remove(filename.split('/')[-1])