from PIL import Image
import os
from os import walk

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
	def __init__(self, from_directory, to_directory):
		self.from_directory = from_directory
		self.to_directory = to_directory
		self.images = []

	"""
	Load the pictures and get their data (filename, picture, sizes)
	"""
	def loadImagesData(self):
		(_, _, yes_filenames) = walk(self.from_directory + 'yes/').next()
		(_, _, no_filenames) = walk(self.from_directory + 'no/').next()

		# Load images here to prevent multiple loadings
		for filename in yes_filenames:
			self.images.append({
				'filename': filename,
				'label': 'yes',
				'image': Image.open(self.from_directory + 'yes/' + filename)
			})

		for filename in no_filenames:
			self.images.append({
				'filename': filename,
				'label': 'no',
				'image': Image.open(self.from_directory + 'no/' + filename)
			})

		# Get max width and max height
		self.getImagesMaxSize()


	"""
	Get the sizes of the largest pictures in both directories
	"""
	def getImagesMaxSize(self):
		self.max_width = 0
		self.max_height = 0

		for im in self.images:
			width, height = im['image'].size
			
			self.max_width = width if width >= self.max_width else self.max_width
			self.max_height = height if height >= self.max_height else self.max_height


	"""
	Resize all images in both directories
	@param mode The resizing mode to use
	@param background_color The color used for padding
	@param shape The shape of the resized picture
	@param square_size The size of the picture if the shape is SHAPE_SQUARE
	"""
	def resizeImages(self, mode, background_color = 'black', shape = SHAPE_SQUARE, square_size = SIZE_AUTO):
		new_images = []

		if shape == self.SHAPE_SQUARE:
			if square_size > 0:
				self.max_width = square_size
				self.max_height = square_size

			else:
				self.max_width = max(self.max_width, self.max_height)
				self.max_height = self.max_width

		
		if mode == self.MODE_PADDING:

			for im in self.images:
				image = im['image']
				new_images.append(im)

				tmp = Image.new(image.mode, (self.max_width, self.max_height), background_color)
				tmp.paste(
					image,
					((self.max_width - image.width)/2, (self.max_height - image.height)/2)
				)

				new_images[-1]['image'] = tmp

			self.images = new_images


		elif mode == self.MODE_RESIZING:

			for im in self.images:
				image = im['image']
				new_images.append(im)

				tmp = image.resize((self.max_width, self.max_height), Image.ANTIALIAS)

				new_images[-1]['image'] = tmp

			self.images = new_images


		elif mode == self.MODE_RESIZING_KEEP_RATIO:

			max_ratio = float(self.max_width) / self.max_height

			for im in self.images:
				image = im['image']
				new_images.append(im)

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
					((self.max_width - tmp_resize.width)/2, (self.max_height - tmp_resize.height)/2)
				)

				new_images[-1]['image'] = tmp

			self.images = new_images


	"""
	Convert pictures in both directories into grayscale mode
	"""
	def convertImages2GrayscaleMode(self):
		new_images = []
		for im in self.images:
			new_images.append(im)
			new_images[-1]['image'] = im['image'].convert('L')

		self.images = new_images


	"""
	Convert pictures in both directories into RGB mode.
	Use after convertImages2RGBMode() to get a grayscale picture in RGB mode.
	"""
	def convertImages2RGBMode(self):
		new_images = []
		for im in self.images:
			new_images.append(im)
			new_images[-1]['image'] = im['image'].convert('RGB')

		self.images = new_images


	"""
	Save all images from both directories to the sub folder ./normalized
	"""
	def saveImages(self):
		# Create the folders if needed
		if not os.path.exists(self.to_directory):
			os.makedirs(self.to_directory)

		if not os.path.exists(self.to_directory + 'yes/'):
			os.makedirs(self.to_directory + 'yes/')

		if not os.path.exists(self.to_directory + 'no/'):
			os.makedirs(self.to_directory + 'no/')


		for im in self.images:
			if im['label'] == 'yes':
				im['image'].save(self.to_directory + 'yes/' + im['filename'], 'JPEG')

			elif im['label'] == 'no':
				im['image'].save(self.to_directory + 'no/' + im['filename'], 'JPEG')



def main():
	
	# Pictures folder
	IMAGES_FROM_DIRECTORY = 'images/'
	IMAGES_TO_DIRECTORY = 'results/'

	# Create the normalizer
	imgn = ImageNormalizer(IMAGES_FROM_DIRECTORY, IMAGES_TO_DIRECTORY)

	# Find all files to manage
	imgn.loadImagesData()

	# Resize images
	imgn.resizeImages(
		# Resizing mode
		# Options are: MODE_PADDING, MODE_RESIZING, MODE_RESIZING_KEEP_RATIO
		mode = ImageNormalizer.MODE_RESIZING_KEEP_RATIO,

		# Background color
		# Can be a string or a hex code
		background_color = 'black',

		# Shape of the normalized image
		# Options are: SHAPE_SQUARE, SHAPE_RECTANGLE
		shape = ImageNormalizer.SHAPE_SQUARE,

		# Size of the picture if square shape is used
		# Can be SIZE_AUTO or a integer (in pixels)
		square_size = ImageNormalizer.SIZE_AUTO
	)

	# Convert all images to grayscale
	imgn.convertImages2GrayscaleMode()

	# Back to RGB mode
	imgn.convertImages2RGBMode()

	# Save normalized images
	imgn.saveImages()




if __name__== "__main__":
	main()