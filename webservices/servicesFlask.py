import flask
from flask import request
from PIL import Image

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Services web disponibles</h1>
<p>Transformer une image en matrice /api/v1/transform?path=[PATH_PICTURE]</p>'''

@app.route('/api/v1/transform', methods=['GET'])
def transform_picture():
    # Check if a PATH was provided as part of the URL.
    # If PATH is provided, assign it to a variable.
    # If no PATH is provided, display an error in the browser.
    if 'path' in request.args:
        path = str(request.args['path'])
    else:
        return "Error: No path field provided. Please specify a path."
    try:
        im = Image.open(path)
        #break
    except IOError:
        return "No picture at the specified path"
    
    data = []
    for i in range (0, im.height):
        for j in range (0, im.width):
            data.append(im.getpixel((j,i)))
        
    return str(data)

app.run()
