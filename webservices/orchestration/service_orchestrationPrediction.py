import flask
from flask import request
from flask_cors import CORS
import os
import json
import pandas as pd

from hdfs import InsecureClient

#from OrchestrationPrediction import OrchestrationPrediction

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

list_algo_deep = ["cnn", "resnet", "alexnet", "vgg"]
list_algo_ml = ["knn", "svm", "gbc", "rfc", "nn"]

# Pour lancer ce service en local, il faut avoir modifié les ports des microservices (par ex, app.run(port = 5001)) car sinon impossible de lancer plusieurs apps flask en même temps
@app.route('/api/v1/orchestrationPrediction', methods=['POST'])
def orchestrationPrediction():

    if request.files['picture'] is None:
        return json.dumps(None)

    hdfs_client = InsecureClient('http://192.168.1.4:9870', user='hadoop')

    picture = request.files['picture']

    with hdfs_client.write('/image_test/test.jpg') as writer:
        picture.save(writer)

    data = pd.DataFrame(['/image_test/test.jpg'], columns=['Path'])
    with hdfs_client.write('/image_test/data.csv', encoding = 'utf-8') as writer:
        data.to_csv(writer, index_label='index')

    classifiers = request.form.getlist('classifiers')

    list_algo = []

    for algo in classifiers:
        if not(algo in list_algo_deep) and not(algo in list_algo_ml):
            return algo + ' is an incorrect algo.'
        list_algo.append(algo)

    orchPred = OrchestrationPrediction('test.jpg', list_algo)
    list_returns_predict = orchPred.run()

    string_result = '{ \"returns_predictions\": {'
    for i in range(len(list_returns_predict)):
        string_result += list_returns_predict[i]
        if i == len(list_returns_predict) - 1:
            string_result += '}}'
        else:
            string_result += ','

    return json.loads(string_result)

app.run(host="0.0.0.0", port = 5013)

#http://localhost:5013/api/v1/orchestrationPrediction?json={%22url_img%22:%22/home/theo/Bureau/INSA/Integrateur/new_data_norm/yes/Y1.jpg%22,%20%22classifiers%22:[%22resnet%22,%20%22alexnet%22]}
