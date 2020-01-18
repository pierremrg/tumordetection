import flask
from flask import request
from dask import dataframe

from MachineLearning import *

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/v1/ml', methods=['GET'])
def machine_learning():
	# Check argument
	if request.args.get('images_directory') is None :
		return 'No "images_directory" given.'

	if request.args.get('algorithm') is None :
		return 'No "algorithm" given.'

	if request.args.get('save_directory') is None :
		return 'No "save_directory" given.'


	images_directory = request.args.get('images_directory')
	algorithm = str(request.args.get('algorithm'))
	save_directory = request.args.get('save_directory')

	# creates new MachineLearning object
	ml = MachineLearning(images_directory, save_directory)
	
	# error detection
	if len(ml.imgs) == 0 or len(ml.labels) == 0:
		app.logger.error("No images were read!")
		return "Error: No images were read!"

	if (algorithm == "knn"):
		ml.knn(ml.imgs, ml.labels)
	elif (algorithm == "svm"):
		ml.svm(ml.imgs, ml.labels)
	elif (algorithm == "gbc"):
		ml.gbc(ml.imgs, ml.labels)
	elif (algorithm == "rfc"):
		ml.rfc(ml.imgs, ml.labels)
	elif (algorithm == "nn"):
		ml.nn(ml.imgs, ml.labels)
	else:
		app.logger.warning("Unexpected algorithm choice, choosing default!")
		ml.svm(ml.imgs, ml.labels)

	return "Model " + str(algorithm) + " trained"

app.run(port = 5007)
