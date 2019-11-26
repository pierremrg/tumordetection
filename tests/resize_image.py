from PIL import Image
from os import walk

class ImageNormalizer:

	MODE_PADDING = 0
	MODE_RESIZING = 1
	MODE_RESIZING_KEEP_RATIO = 2

	def __init__(self, directory):
		self.directory = directory
		self.images = []


	def loadImagesData(self):
		(dirpath, dirnames, filenames) = walk(self.directory).next()
		self.filenames = [self.directory + filename for filename in filenames]

		# Load images here to prevent multiple loadings
		for filename in self.filenames:
			self.images.append({
				'filename': filename,
				'image': Image.open(filename)
			})

		# Get max width and max height
		self.getImagesMaxSize()


	def getImagesMaxSize(self):
		self.max_width = 0
		self.max_height = 0

		for im in self.images:
			width, height = im['image'].size
			
			self.max_width = width if width >= self.max_width else self.max_width
			self.max_height = height if height >= self.max_height else self.max_height


	def resizeImages(self, mode, color = 'black'):
		self.max_width
		self.max_height

		new_images = []

		if mode == self.MODE_PADDING:

			for im in self.images:
				image = im['image']
				new_images.append(im)

				tmp = Image.new(image.mode, (self.max_width, self.max_height), color)
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

			for im in self.images:
				image = im['image']
				new_images.append(im)

				# Get right width and height
				new_height = image.height * self.max_width / image.width # 4000

				new_width = image.width * self.max_height / image.height # 500

				if new_height > self.max_height:
					# Keep new_width
					


				if image.width < image.height:
					new_width = self.max_width
					new_height = image.height * new_width / image.width

				else:
					new_height = self.max_height
					new_width = image.width * new_height / image.height

				tmp = image.resize((new_width, new_height), Image.ANTIALIAS)

				new_images[-1]['image'] = tmp

			self.images = new_images



	def convertImagesToGrayscale(self):
		new_images = []
		for im in self.images:
			new_images.append(im)
			new_images[-1]['image'] = im['image'].convert('L')

		self.images = new_images


	def saveImages(self):
		i = 0
		for im in self.images:
			im['image'].save(self.filenames[i] + ".gray", "JPEG")
			i += 1

 


def main():
	imgn = ImageNormalizer('images/')

	# Find all files to manage
	imgn.loadImagesData()

	# Resize images
	imgn.resizeImages(ImageNormalizer.MODE_RESIZING_KEEP_RATIO, 'white')

	# Convert all images to grayscale
	# imgn.convertImagesToGrayscale()

	# Save normalized images
	imgn.saveImages()



	# normalize(im, 0, 0, 0, 0, 0)

	


if __name__== "__main__":
	main()