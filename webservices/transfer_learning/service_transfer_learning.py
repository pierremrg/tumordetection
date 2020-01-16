import os
import flask
from flask import request

from ModelTransferLearning import ModelTransferLearning

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/v1/transferlearning', methods=['GET'])
def train_transfer_learning():
    if request.args.get('images_directory') is None :
        return 'No "images_directory" given.'
    
    if request.args.get('save_directory') is None :
        return 'No "save_directory" given.'

    images_directory = request.args.get('images_directory')
    save_directory = request.args.get('save_directory')
    
    if not os.path.exists(images_directory):
        return '"images_directory" cannot be found.'

    if request.args.get('batch_size') is not None:
        batch_size = int(request.args.get('batch_size'))
    else:
        batch_size = 30
    
    if request.args.get('network') is not None:
        network = request.args.get('network')
    else:
        network = "resnet"
    
    mtl = ModelTransferLearning(images_directory, save_directory, batch_size, network)

    mtl.run()

    train_accuracy, val_accuracy = cmc.get_accuracy()

    return 'Model ' + str(network) + ' trained : train_accuracy = ' + str(train_accuracy) + ' val_accuracy = ' + str(val_accuracy)

app.run()
