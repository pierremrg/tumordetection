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
	if request.args.get('directory_algo') is None:
		return 'No "directory_algo" given.'
	else:
		directory_algo = request.args.get('directory_algo')

	if request.args.get('algo') is None:
		return 'No "algo" given.'
	elif (not(request.args.get('algo') in list_algo_deep) and not(request.args.get('algo') in list_algo_ml)):
		return 'Incorrect "algo" provided.'
	else:
		algo = request.args.get('algo')

	if request.args.get('path_img') is None:
		return 'No "path_img" given.'
	else:
		path_img = request.args.get('path_img')

	# type algo choisi
	if algo in list_algo_deep:
		pred = Prediction(directory_algo, algo, path_img)
		label, proba = pred.run()

		if label == -1:
			return 'Error model '+str(algo)+' is not trained yet!\nTrain this model first before using it for predictions'
		else:
			#return 'Prediction using ' + str(algo) + ' done, Label found : ' + str(label) + ' - Probability : ' + str(proba)
			return '\"' + str(algo) + '\":{\"label\":'+str(label)+' ,\"proba\":'+str(proba)+'}'

	elif algo in list_algo_ml:
		pred = Prediction_ML(directory_algo, algo, path_img)
		label, proba = pred.run()
		
		if label == -1:
			return 'Error model '+str(algo)+' is not trained yet!\nTrain this model first before using it for predictions'
		else:
			#return 'Prediction using ' + str(algo) + ' done, Label found : ' + str(label)
			return '\"' + str(algo) + '\":{\"label\":'+str(label)+' ,\"proba\":'+str(proba)+'}'

	else:
		return 'Unexpected error, ' + str(algo) + ' is not part of the supported algorithms!'

app.run(host="0.0.0.0", port=5007)
