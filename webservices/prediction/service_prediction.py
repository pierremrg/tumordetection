import flask
from flask import request
import os

from Prediction import Prediction
from Prediction_ML import Prediction_ML

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# 4 algos deep learning
# 5 algos sklearn
list_algo_deep = ["cnn", "resnet", "alexnet", "vgg"]
list_algo_ml = ["knn", "svm", "gbc", "rfc", "nn"]

@app.route('/api/v1/prediction', methods=['GET'])
def prediction():
	if request.args.get('directory_from') is None:
		return 'No "directory_from" given.'
	else:
		directory_from = request.args.get('directory_from')

	if not os.path.exists(directory_from):
		return '"directory_from" cannot be found.'

	if request.args.get('algo') is None:
		return 'No "algo" given.'
	elif (not(request.args.get('algo') in list_algo_deep) and not(request.args.get('algo') in list_algo_ml)):
		return 'Incorrect "algo" provided.'
	else:
		algo = request.args.get('algo')

	if request.args.get('directory_img') is None:
		return 'No "directory_img" given.'
	else:
		directory_img = request.args.get('directory_img')

	if not os.path.exists(directory_img):
		return '"directory_img" cannot be found.'

	# if request.args.get('name_img') is None:
		# return 'No "name_img" given.'
	# else:
		# name_img = request.args.get('name_img')

	# if not os.path.exists(directory_img + name_img):
		# return 'img ' + name_img + ' cannot be found.'

	# type algo choisi
	if algo in list_algo_deep:
		pred = Prediction(directory_from, algo, directory_img)
		label, proba = pred.run()

		if label == -1:
			return 'Error model '+str(algo)+' is not trained yet!\nTrain this model first before using it for predictions'
		else:
			return 'Prediction using ' + str(algo) + ' done, Label found : ' + str(label) + ' - Probability : ' + str(proba)

	elif algo in list_algo_ml:
		pred = Prediction_ML(directory_from, algo, directory_img)
		label = pred.run()
		
		if label == -1:
			return 'Error model '+str(algo)+' is not trained yet!\nTrain this model first before using it for predictions'
		else:
			return 'Prediction using ' + str(algo) + ' done, Label found : ' + str(label)

	else:
		return 'Unexpected error, ' + str(algo) + ' is not part of the supported algorithms!'

app.run(port=5007)
