import flask
from flask import request
from PIL import Image
import os
from os import walk
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client

from ImageNormalizer import ImageNormalizer

app = flask.Flask(__name__)
app.config["DEBUG"] = True

client = Client('127.0.0.1:50190')
print(client)

def create_csv(directory_to):
    list_yes = os.listdir(directory_to + 'yes')
    list_images = ['yes/' + name for name in list_yes]
    list_no = os.listdir(directory_to + 'no/')
    list_images += ['no/' + name for name in list_no]

    data = pd.DataFrame(list_images, columns=['Path'])
    data.to_csv(directory_to + 'data.csv', index_label='index')


def compute_norm(df, from_dir, out_dir):
	print('toto')
	# Create the normalizer
	imgn = ImageNormalizer(from_dir, df.Path, out_dir)

	# Find all files to manage
	# imgn.loadImagesData()

	# # Resize images
	# imgn.resizeImages(
	#     # Resizing mode
	#     # Options are: MODE_PADDING, MODE_RESIZING, MODE_RESIZING_KEEP_RATIO
	#     mode = ImageNormalizer.MODE_RESIZING_KEEP_RATIO,

	#     # Background color
	#     # Can be a string or a hex code
	#     background_color = 'black',

	#     # Shape of the normalized image
	#     # Options are: SHAPE_SQUARE, SHAPE_RECTANGLE
	#     shape = 1000,

	#     # Size of the picture if square shape is used
	#     # Can be SIZE_AUTO or a integer (in pixels)
	#     square_size = ImageNormalizer.SIZE_AUTO
	# )

	# # Convert all images to grayscale
	# imgn.convertImages2GrayscaleMode()

	# # Back to RGB mode
	# imgn.convertImages2RGBMode()

	# # Save normalized images
	# imgn.saveImages()


@app.route('/api/v1/normalize', methods=['POST'])
def normalize():

	# Check pictures folders
	if request.args.get('from') is None:
		return 'No "from" directory given.'

	if request.args.get('to') is None:
		return 'No "to" directory given.'

	if not os.path.exists(request.args.get('from')):
		return '"from" directory cannot be found.'

	data = dd.read_csv(request.args.get('from') + 'data.csv')
	data = data.repartition(npartitions=8)

	data.map_partitions(compute_norm,
						request.args.get('from'),
						request.args.get('to'),
						meta='dask.dataframe.core.Series').compute()
	
	#create_csv(directory_to=request.args.get('to'))

	return 'Normalization done.'

app.run()
