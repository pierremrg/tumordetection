import flask
from flask import request
from dask import dataframe

from MachineLearning import *

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/v1/ml', methods=['GET'])
def machine_learning():
	# Check argument
	if request.args.get('directory_from') is None :
		return 'No "directory_from" given.'

	if request.args.get('path_image') is None :
		return 'No "path_image" given.'

	if request.args.get('algorithm') is None :
		return 'No "algorithm" given.'

	if request.args.get('fast_train') is None :
		return 'No "fast_train" given.'

	if request.args.get('path_algorithm') is None :
		return 'No "path_algorithm" given.'


	directory_from = request.args.get('directory_from')
	path_image = request.args.get('path_image')
	algorithm = str(request.args.get('algorithm'))
	fast_train = bool(request.args.get('fast_train'))
	path_algorithm = request.args.get('path_algorithm')

	# creates new MachineLearning object
	ml = MachineLearning(directory_from, path_image, path_algorithm)
	
	# error detection
	if len(ml.imgs) == 0 or len(ml.labels) == 0:
		app.logger.error("No images were read!")
		return "Error: No images were read!"

	if (algorithm == "knn"):
		res = ml.knn(ml.img, ml.imgs, ml.labels, fast_train)
	elif (algorithm == "svm"):
		res = ml.svm(ml.img, ml.imgs, ml.labels, fast_train)
	elif (algorithm == "gbc"):
		res = ml.gbc(ml.img, ml.imgs, ml.labels, fast_train)
	elif (algorithm == "rfc"):
		res = ml.rfc(ml.img, ml.imgs, ml.labels, fast_train)
	elif (algorithm == "nn"):
		res = ml.gbc(ml.img, ml.imgs, ml.labels, fast_train)
	else:
		app.logger.warning("Unexpected algorithm choice, choosing default!")
		res = ml.svm(ml.img, ml.imgs, ml.labels, fast_train)

	app.logger.info("Prediction: "+str(res))
	return str(res)

app.run()