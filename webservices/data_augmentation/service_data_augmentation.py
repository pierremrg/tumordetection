import flask
from flask import request
import os
import pandas as pd
from hdfs import InsecureClient

from DataAugmentation import DataAugmentation

app = flask.Flask(__name__)
app.config["DEBUG"] = True

def create_csv(directory_to):
    hdfs_client = InsecureClient('http://192.168.1.4:9870', user='hadoop')
    list_yes = hdfs_client.list('/' + directory_to + 'yes')
    list_images = ['yes/' + name for name in list_yes]
    list_no = hdfs_client.list('/' + directory_to + 'no/')
    list_images += ['no/' + name for name in list_no]

    data = pd.DataFrame(list_images, columns=['Path'])

    with hdfs_client.write('/' + directory_to + 'data.csv', encoding = 'utf-8') as writer:
    	data.to_csv(writer, index_label='index')

@app.route('/api/v1/data_augment', methods=['POST'])
def data_augment() :
    # Check argument
    if request.args.get('directory_from') is None :
        return 'No "directory_from" given.'

    if request.args.get('max_augmentation') is None :
        return 'No "max_augmentation" given.'

    directory_from = request.args.get('directory_from')
    max_augmentation =  int(request.args.get('max_augmentation'))

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
