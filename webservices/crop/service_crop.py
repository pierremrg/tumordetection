import flask
from flask import request
import os
import pandas as pd
import dask
import dask.dataframe as dd
from dask.distributed import Client

from ImageCropper import ImageCropper

app = flask.Flask(__name__)
app.config["DEBUG"] = True

client = Client('127.0.0.1:53880')

def create_csv(directory_to):
    list_yes = os.listdir(directory_to + 'yes')
    list_images = ['yes/' + name for name in list_yes]
    list_no = os.listdir(directory_to + 'no/')
    list_images += ['no/' + name for name in list_no]

    data = pd.DataFrame(list_images, columns=['Path'])
    data.to_csv(directory_to + 'data.csv', index_label='index')

def compute_crop(df, from_dir, out_dir):
	DIRS = ["yes", "no"]

	# creates the cropper object
	imgc = ImageCropper(from_dir, df.Path, out_dir, DIRS)
	
	# creates the output folders
	imgc.createOutputDirectory()
	
	# does the job
	imgc.cropImages()

@app.route('/api/v1/crop', methods=['GET'])
def crop():
	# Check pictures folders
	if request.args.get('from') is None:
		return 'No "from" directory given.'

	if request.args.get('to') is None:
		return 'No "to" directory given.'

	if not os.path.exists(request.args.get('from')):
		return '"from" directory cannot be found.'
		
	data = dd.read_csv(request.args.get('from') + 'data.csv')
	data = data.repartition(npartitions=8)

	data.map_partitions(compute_crop,
						request.args.get('from'),
						request.args.get('to'),
						meta='dask.dataframe.core.Series').compute()
	
	create_csv(directory_to=request.args.get('to'))
	
	return "Crop finished"

app.run()
