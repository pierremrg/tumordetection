import flask
from flask import request
from flask_cors import CORS
import json
import time

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/v1/simu_orchestrateur', methods=['POST'])
def normalize():

    time.sleep(2)

    if request.files['picture'] is None:
        return json.dumps(None)

    print(request.form.getlist('classifiers'))

    data = {
        'returns_predictions': {
            'alexnet': {
                'label': 1,
                'proba': 0.98
            },
            'resnet': {
                'label': 1,
                'proba': 0.91,
            },
            'knn': {
                'label': 0,
                'proba': 0.42
            }
        }
    }

    return json.dumps(data)

app.run(host="0.0.0.0", port="5015")
