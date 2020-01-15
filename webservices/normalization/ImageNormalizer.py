from PIL import Image
import os
from os import walk
from hdfs import InsecureClient

class ImageNormalizer:

	MODE_PADDING = 0
	MODE_RESIZING = 1
	MODE_RESIZING_KEEP_RATIO = 2

	SHAPE_SQUARE = 0
	SHAPE_RECTANGLE = 1

	SIZE_AUTO = -1

	"""
	Create a new image normalizer

	@params from_directory The pictures folder. Two sub-folders should be inside:
		one called "yes" and another one called "no"
	@params to_directory The destination folder, where to solve new pictures
	"""
	def __init__(self, from_directory, images_path, to_directory):
		self.from_directory = from_directory
		self.to_directory = to_directory
		self.images_path = images_path
		self.hdfs_client = InsecureClient('http://192.168.1.4:9870', user='hadoop')

	"""
	Resize all images in both directories
	@param mode The resizing mode to use
	@param background_color The color used for padding
	@param shape The shape of the resized picture
	@param square_size The size of the picture if the shape is SHAPE_SQUARE
	"""
	def resizeImages(self, mode, background_color = 'black', shape = SHAPE_SQUARE, square_size = 1000, to_grayscale = True):
		if shape == self.SHAPE_SQUARE:
			self.max_width = square_size
			self.max_height = square_size
		
		if mode == self.MODE_PADDING:

			for image_name in self.images_path:
				with self.hdfs_client.read('/' + self.from_directory + image_name) as reader:        
					image = Image.open(reader)

				tmp = Image.new(image.mode, (self.max_width, self.max_height), background_color)
				tmp.paste(
					image,
					(int((self.max_width - image.width))/2, int((self.max_height - image.height)/2))
				)

				tmp = self.convertImage(tmp, to_grayscale)

				with self.hdfs_client.write('/' + self.to_directory + image_name) as writer:
					tmp.save(writer, format='jpeg')


		elif mode == self.MODE_RESIZING:

			for image_name in self.images_path:
				with self.hdfs_client.read('/' + self.from_directory + image_name) as reader:        
					image = Image.open(reader)

				tmp = image.resize((self.max_width, self.max_height), Image.ANTIALIAS)

				tmp = self.convertImage(tmp, to_grayscale)

				with self.hdfs_client.write('/' + self.to_directory + image_name) as writer:
					tmp.save(writer, format='jpeg')


		elif mode == self.MODE_RESIZING_KEEP_RATIO:

			max_ratio = float(self.max_width) / self.max_height

			for image_name in self.images_path:
				with self.hdfs_client.read('/' + self.from_directory + image_name) as reader:        
					image = Image.open(reader)

				# Get right width and height
				ratio = float(image.width) / image.height

				if ratio > max_ratio:
					# width = width max
					new_width = self.max_width
					new_height = new_width / ratio

				else:
					# height = height max
					new_height = self.max_height
					new_width = new_height * ratio


				tmp_resize = image.resize((int(new_width), int(new_height)), Image.ANTIALIAS)

				tmp = Image.new(image.mode, (self.max_width, self.max_height), background_color)
				tmp.paste(
					tmp_resize,
					(int((self.max_width - tmp_resize.width)/2), int((self.max_height - tmp_resize.height)/2))
				)

				tmp = self.convertImage(tmp, to_grayscale)

				with self.hdfs_client.write('/' + self.to_directory + image_name) as writer:
					tmp.save(writer, format='jpeg')

	def convertImage(self, image, to_grayscale):
		if (to_grayscale):
			image = image.convert('L')

		return image.convert('RGB')
