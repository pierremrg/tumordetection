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

    if request.args.get('directory_to') is None :
        return 'No "directory_to" given.'

    directory_from = request.args.get('directory_from')
    max_augmentation =  int(request.args.get('max_augmentation'))
    directory_to = request.args.get('directory_to')

    if request.args.get('coef_rotation') is not None:
        coef_rotation = float(request.args.get('coef_rotation'))
    else:
        coef_rotation = 0.7

    dtaug = DataAugmentation(directory_from, max_augmentation, coef_rotation, directory_to)

    dtaug.run()

    create_csv(directory_to=directory_to)

    return 'Data augmentation done'

app.run(host="0.0.0.0", port="5002")
