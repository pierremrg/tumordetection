import flask, os
import pandas as pd
from flask import request
import logging
import hdfs
import shutil

from PictureGetter import PictureGetter

app = flask.Flask(__name__)
app.config["DEBUG"] = True

def create_csv(directory_to):
    list_yes = os.listdir(directory_to + 'yes')
    list_images = ['yes/' + name for name in list_yes]
    list_no = os.listdir(directory_to + 'no/')
    list_images += ['no/' + name for name in list_no]

    data = pd.DataFrame(list_images, columns=['Path'])
    data.to_csv(directory_to + 'data.csv', index_label='index')

def upload_files_to_hdfs(directory_to):
    client = hdfs.InsecureClient('http://192.168.1.4:9870', user='hadoop')
    client.upload('/' + directory_to, 'tmp/')
    shutil.rmtree('tmp/')

@app.route('/api/v1/getPictures', methods=['POST'])
def get_pictures():
    if request.args.get('directory_to') is None :
        return 'No "directory_to" given.'

    if request.args.get('url') is None :
        return 'No "url" given.'
    
    directory_to = request.args.get('directory_to')
    url = request.args.get('url')

    getPic = PictureGetter(url=url, directory_to=directory_to)
    getPic.run()

    create_csv('tmp/')

    upload_files_to_hdfs(directory_to)

    return 'Images téléchargées'

app.run(host="0.0.0.0", port=5001)

