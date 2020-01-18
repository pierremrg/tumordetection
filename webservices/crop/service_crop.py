import numpy as np
import os
import pandas as pd
import imutils
import cv2
import sys
import logging

import flask
from flask import request

import dask
import dask.dataframe as dd
from dask.distributed import Client

from hdfs import InsecureClient
from ImageCropper import ImageCropper

app = flask.Flask(__name__)
app.config["DEBUG"] = True

formatter = logging.Formatter(fmt = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

def create_csv(directory_to):
    hdfs_client = InsecureClient('http://192.168.1.4:9870', user='hadoop')
    list_yes = hdfs_client.list('/' + directory_to + 'yes')
    list_images = ['yes/' + name for name in list_yes]
    list_no = hdfs_client.list('/' + directory_to + 'no/')
    list_images += ['no/' + name for name in list_no]

    data = pd.DataFrame(list_images, columns=['Path'])

    with hdfs_client.write('/' + directory_to + 'data.csv', encoding = 'utf-8') as writer:
    	data.to_csv(writer, index_label='index')

def compute_crop(df, input_folder, output_folder):

	imgc = ImageCropper(input_folder, df.Path, output_folder)

	imgc.cropImages()

@app.route('/api/v1/crop', methods=['GET'])
def crop():
	# Check pictures folders
	if request.args.get('from') is None:
		return 'No "from" directory given.'

	if request.args.get('to') is None:
		return 'No "to" directory given.'

	directory_from = request.args.get('from')
	directory_to = request.args.get('to')

	dask_client = Client('192.168.1.4:8786')
	hdfs_client = InsecureClient('http://192.168.1.4:9870', user='hadoop')	

	with hdfs_client.read('/' + directory_from + 'data.csv') as reader:
		data = pd.read_csv(reader)
		data = dd.from_pandas(data, npartitions=24)

	data.map_partitions(compute_crop,
						directory_from,
						directory_to,
						meta='dask.dataframe.core.Series').compute()
	
	create_csv(directory_to=directory_to)
	
	return "Crop finished"

app.run(host="0.0.0.0", port="5004")
