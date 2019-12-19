import wget, zipfile, flask, os
from flask import request
import logging

from PictureGetter import PictureGetter

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/v1/getPictures', methods=['GET'])
def get_pictures():
    getPic = PictureGetter()
    getPic.run()
    return 'Images téléchargées'

app.run()

