import flask
from flask import request
import os

from DataAugmentation import DataAugmentation

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/v1/data_augment', methods=['POST'])
def data_augment() :
    # Check argument
    if request.args.get('directory_from') is None :
        return 'No "directory_from" given.'

    if request.args.get('max_augmentation') is None :
        return 'No "max_augmentation" given.'

    directory_from = request.args.get('directory_from')
    max_augmentation =  int(request.args.get('max_augmentation'))

    if not os.path.exists(directory_from):
        return '"directory_from" cannot be found.'

    if request.args.get('coef_rotation') is not None:
        coef_rotation = float(request.args.get('coef_rotation'))
    else:
        coef_rotation = 0.7

    if request.args.get('directory_to') is not None :
        directory_to = request.args.get('directory_to')

        if not os.path.exists(directory_to):
            return '"directory_to" cannot be found.'
    else:
        directory_to = None

    dtaug = DataAugmentation(directory_from, max_augmentation, coef_rotation, directory_to)

    dtaug.run()

    return 'Data augmentation done'

app.run()
