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

from hdfs import InsecureClient

app = flask.Flask(__name__)
app.config["DEBUG"] = True

def create_csv(directory_to):
	hdfs_client = InsecureClient('http://192.168.1.4:9870', user='hadoop')
	# Cas où on entraine
	if (len(hdfs_client.list('/' + directory_to)) != 1):
		list_yes = hdfs_client.list('/' + directory_to + 'yes')
		list_images = ['yes/' + name for name in list_yes]
		list_no = hdfs_client.list('/' + directory_to + 'no/')
		list_images += ['no/' + name for name in list_no]
	# Cas où on prédit, on a une image
	else:
		list_images = hdfs_client.list('/' + directory_to)

	data = pd.DataFrame(list_images, columns=['Path'])

	with hdfs_client.write('/' + directory_to + 'data.csv', encoding = 'utf-8') as writer:
		data.to_csv(writer, index_label='index')


def compute_norm(df, from_dir, out_dir):
	# Create the normalizer
	imgn = ImageNormalizer(from_dir, df.Path, out_dir)

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
		square_size = 1000,

		to_grayscale = True
	)

@app.route('/api/v1/normalize', methods=['POST'])
def normalize():

	# Check pictures folders
	if request.args.get('from') is None:
		return 'No "from" directory given.'

	if request.args.get('to') is None:
		return 'No "to" directory given.'

	dask_client = Client('192.168.1.4:8786')
	hdfs_client = InsecureClient('http://192.168.1.4:9870', user='hadoop')	

	from_directory = request.args.get('from')
	to_directory = request.args.get('to')

	with hdfs_client.read('/' + from_directory + 'data.csv') as reader:
		data = pd.read_csv(reader)
		data = dd.from_pandas(data, npartitions=24)

	data.map_partitions(compute_norm,
						from_directory,
						to_directory,
						meta='dask.dataframe.core.Series').compute()
	
	create_csv(directory_to=to_directory)

	return 'Normalization done.'

app.run(host="0.0.0.0", port="5003")
