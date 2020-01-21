import flask
from flask import request
from dask import dataframe

from MachineLearning import *

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/v1/ml', methods=['GET'])
def machine_learning():
	algo_list = ["knn", "svm", "gbc", "rfc", "nn"]

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
	if algorithm == "nn":
		ml = MachineLearning(images_directory, save_directory, 32)
	else:
		ml = MachineLearning(images_directory, save_directory)
	
	# error detection
	if len(ml.imgs) == 0 or len(ml.labels) == 0:
		app.logger.error("No images were read!")
		return "Error: No images were read!"

	if algorithm in algo_list:
		score_train, score_test = ml.train(algorithm, ml.imgs, ml.labels)
	else:
		app.logger.warning("Unexpected algorithm choice, choosing default!")
		algorithm = "svm"
		score_train, score_test = ml.train(algorithm, ml.imgs, ml.labels)

	return '\"' + algorithm + '\":{\"train_acc\":'+str(score_train)+' ,\"val_acc\":'+str(score_test)+'}'

app.run(port = 5007)
