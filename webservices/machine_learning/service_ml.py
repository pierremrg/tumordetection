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

    if request.args.get('image') is None :
        return 'No "image" given.'

    if request.args.get('algorithm') is None :
        return 'No "algorithm" given.'

    if request.args.get('fast_train') is None :
        return 'No "fast_train" given.'

    directory_from = request.args.get('directory_from')
	image = request.args.get('image')
    algorithm = int(request.args.get('algorithm'))
	fast_train = bool(request.args.get('fast_train'))

	# creates new MachineLearning object
	ml = MachineLearning(directory_from)
	
	# error detection
	if len(ml.imgs) == 0 or len(ml.labels) == 0:
		logging.error("No images were read!")
		return "Error: No images were read!"

	if (algorithm == "knn"):
		res = ml.knn(image, ml.imgs, ml.labels, fast_train)
	elif (algorithm == "svm"):
		res = ml.svm(image, ml.imgs, ml.labels, fast_train)
	elif (algorithm == "gbc"):
		res = ml.gbc(image, ml.imgs, ml.labels, fast_train)
	elif (algorithm == "rfc"):
		res = ml.rfc(image, ml.imgs, ml.labels, fast_train)
	elif (algorithm == "nn"):
		res = ml.gbc(image, ml.imgs, ml.labels, fast_train)
	else:
		logging.warning("Unexpected algorithm choice, choosing default!")
		res = ml.svm(image, ml.imgs, ml.labels, fast_train)

	return res

app.run()