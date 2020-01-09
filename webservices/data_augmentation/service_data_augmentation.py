import flask
from flask import request
import os
import pandas as pd

from DataAugmentation import DataAugmentation

app = flask.Flask(__name__)
app.config["DEBUG"] = True

def create_csv(directory_to):
    list_yes = os.listdir(directory_to + 'yes')
    list_images = ['yes/' + name for name in list_yes]
    list_no = os.listdir(directory_to + 'no/')
    list_images += ['no/' + name for name in list_no]

    data = pd.DataFrame(list_images, columns=['Path'])
    data.to_csv(directory_to + 'data.csv', index_label='index')

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
    else:
        directory_to = None

    dtaug = DataAugmentation(directory_from, max_augmentation, coef_rotation, directory_to)

    dtaug.run()

    if (directory_to != None):
        save_dir = directory_to
    else:
        save_dir = directory_from

    create_csv(directory_to=save_dir)

    return 'Data augmentation done'

app.run()
