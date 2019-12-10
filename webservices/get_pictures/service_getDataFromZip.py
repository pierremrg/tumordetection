import wget, zipfile, flask, os
from flask import request
import logging

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/v1/getPictures', methods=['GET'])
def get_pictures():
    logging.basicConfig(level = logging.DEBUG)
    url = 'https://docs.google.com/uc?export=download&id=1ZiT1jsKT6WQ0UwD3eGKOCxpekSY2iVQV'
    wget.download(url, '../../')
    logging.info('Zip téléchargé')
    with zipfile.ZipFile('../../irm_images.zip', 'r') as zip_ref:
        zip_ref.extractall('../../')
        logging.info('Dossier dézippé')
    #Au lieu de les mettre sur un dossier il faudra les mettre dans la BDD quand elle sera créée
    os.remove("../../irm_images.zip")
    return 'Images téléchargées'

app.run()

