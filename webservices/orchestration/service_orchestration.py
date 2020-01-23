import flask
from flask import request
from flask_cors import CORS
import os
import json

from Orchestration import Orchestration

from hdfs import InsecureClient

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

list_algo_deep = ["cnn", "resnet", "alexnet", "vgg"]
list_algo_ml = ["knn", "svm", "gbc", "rfc", "nn"]

# Pour lancer ce service en local, il faut avoir modifié les ports des microservices (par ex, app.run(port = 5001)) car sinon impossible de lancer plusieurs apps flask en même temps
@app.route('/api/v1/orchestrationTraining', methods=['POST'])
def orchestrationTraining():
    hdfs_cli = InsecureClient('http://192.168.1.4:9870', user='hadoop')

    hdfs_cli.delete('/images', recursive=True)
    hdfs_cli.delete('/images_augmented', recursive=True)
    hdfs_cli.delete('/images_crop', recursive=True)
    hdfs_cli.delete('/images_norm', recursive=True)

    data = request.get_json()
    
    url = data["url_db"]

    classifiers = data["classifiers"]


    list_algo = []

    for algo in classifiers:
        if not(algo in list_algo_deep) and not(algo in list_algo_ml):
            return algo + ' is an incorrect algo.'
        list_algo.append(algo)

    orch = Orchestration(url, list_algo)
    list_returns_trains = orch.run()

    string_result = '{ \"returns_trains\": {'
    for i in range(len(list_returns_trains)):
        string_result += list_returns_trains[i]
        if i == len(list_returns_trains) - 1:
            string_result += '}}'
        else:
            string_result += ','
    return json.loads(string_result)

app.run(host="0.0.0.0", port = 5009)

#http://localhost:5006/api/v1/orchestrationTraining?json={%22url_db%22:%22aaaa%22,%20%22classifiers%22:[%22resnet%22},%22knn%22]}
