import os
import flask
from flask import request

from ClassMediumCNN import ClassMediumCNN

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/v1/medium_cnn', methods=['POST'])
def train_medium_cnn():
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
    
    if request.args.get('epochs') is not None:
        epochs = int(request.args.get('epochs'))
    else:
        epochs = 30
    
    cmc = ClassMediumCNN(images_directory, save_directory, batch_size, epochs)

    cmc.run()

    train_accuracy, val_accuracy = cmc.get_accuracy()

    return 'Medium CNN trained : train_accuracy = ' + str(train_accuracy) + ' val_accuracy = ' + str(val_accuracy)

app.run()
