import flask
from flask import request
from PIL import Image
import os
from os import walk

from ImageNormalizer import ImageNormalizer

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/v1/normalize', methods=['POST'])
def normalize():

	# Check pictures folders
	if request.args.get('from') is None:
		return 'No "from" directory given.'

	if request.args.get('to') is None:
		return 'No "to" directory given.'

	if not os.path.exists(request.args.get('from')):
		return '"from" directory cannot be found.'


	# Create the normalizer
	imgn = ImageNormalizer(request.args.get('from'), request.args.get('to'))

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


	return 'Normalization done.'

app.run(host="0.0.0.0")
