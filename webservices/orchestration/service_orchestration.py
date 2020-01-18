import flask
from flask import request
import os
import json

from Orchestration import Orchestration

app = flask.Flask(__name__)
app.config["DEBUG"] = True

list_algo_deep = ["cnn", "resnet", "alexnet", "vgg"]
list_algo_ml = ["knn", "svm", "gbc", "rfc", "nn"]

# Pour lancer ce service en local, il faut avoir modifié les ports des microservices (par ex, app.run(port = 5001)) car sinon impossible de lancer plusieurs apps flask en même temps
@app.route('/api/v1/orchestrationTraining', methods=['GET'])
def orchestrationTraining():

    #JSON (url_db & classifiers)
    if request.args.get('json') is None:
        return 'No "json" given.'
    else:
        data = request.args.get('json')

    data = json.loads(data)
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

app.run(port = 5009)

#http://localhost:5006/api/v1/orchestrationTraining?json={%22url_db%22:%22aaaa%22,%20%22classifiers%22:[%22resnet%22},%22knn%22]}