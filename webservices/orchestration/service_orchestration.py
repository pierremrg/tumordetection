import flask
from flask import request
import os

from Orchestration import Orchestration

app = flask.Flask(__name__)
app.config["DEBUG"] = True

list_algo = ["cnn", "resnet", "alexnet", "vgg"]

# Pour lancer ce service en local, il faut avoir modifié les ports des microservices (par ex, app.run(port = 5001)) car sinon impossible de lancer plusieurs apps flask en même temps
@app.route('/api/v1/orchestration_preprocessing', methods=['POST'])
def orchestration() :

    if request.args.get('directory_to') is None :
        return 'No "directory_to" given.'
    else:
        directory_to = request.args.get('directory_to')

    if request.args.get('max_augmentation') is None :
        return 'No "max_augmentation" given.'

    max_augmentation =  int(request.args.get('max_augmentation'))

    if request.args.get('coef_rotation') is not None:
        coef_rotation = float(request.args.get('coef_rotation'))
    else:
        coef_rotation = 0.7

    if request.args.get('algo') is None :
        return 'No "algo" given.'
    elif not(request.args.get('algo') in list_algo) :
        return 'Incorrect "algo" provided.'
    else:
        algo = request.args.get('algo')

    if request.args.get('batch_size') is not None:
        batch_size = int(request.args.get('batch_size'))
    else:
        batch_size = 30

    if request.args.get('epochs') is not None:
        epochs = int(request.args.get('epochs'))
    else:
        epochs = 30

    orch = Orchestration(directory_to, max_augmentation, coef_rotation, algo, batch_size, epochs)

    orch.run()

    return 'Orchestration done'

app.run(port = 5006)