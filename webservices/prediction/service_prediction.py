import flask
from flask import request
import os

from Prediction import Prediction

app = flask.Flask(__name__)
app.config["DEBUG"] = True

list_algo = ["cnn", "resnet", "alexnet", "vgg"]


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
    elif not(request.args.get('algo') in list_algo):
        return 'Incorrect "algo" provided.'
    else:
        algo = request.args.get('algo')

    if request.args.get('directory_img') is None:
        return 'No "directory_img" given.'
    else:
        directory_img = request.args.get('directory_img')

    if not os.path.exists(directory_img):
        return '"directory_img" cannot be found.'

    if request.args.get('name_img') is None:
        return 'No "name_img" given.'
    else:
        name_img = request.args.get('name_img')

    if not os.path.exists(directory_img + name_img):
        return 'img ' + name_img + ' cannot be found.'

    pred = Prediction(directory_from, algo, directory_img, name_img)

    label, proba = pred.run()

    return 'Prediction done, Label found : ' + str(label) + ' - Probability : ' + str(proba)


app.run(port=5007)
