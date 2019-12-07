import flask
from flask import request
import os

from ImageCropper import ImageCropper

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/v1/crop', methods=['GET'])
def crop():
	print("Toto:", request.args.get('test') )

	# Check pictures folders
	if request.args.get('from') is None:
		return 'No "from" directory given.'

	if request.args.get('to') is None:
		return 'No "to" directory given.'

	if not os.path.exists(request.args.get('from')):
		return '"from" directory cannot be found.'

	DIRS = ["yes", "no"]

	# creates the cropper object
	imgc = ImageCropper(request.args.get('from'), request.args.get('to'), DIRS)
	
	# creates the output folders
	imgc.createOutputDirectory()
	
	# does the job
	imgc.cropImages()
	
	return "Crop finished"

app.run()
