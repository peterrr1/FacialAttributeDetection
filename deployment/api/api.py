from flask import Flask, json, request, jsonify
from model import predict
import os
import urllib.request
from werkzeug.utils import secure_filename
from flask_cors import CORS 

UPLOAD_FOLDER = '/storage'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


app = Flask(__name__, static_folder='../build', static_url_path='/')

CORS(app, supports_credentials=True)
 
app.secret_key = "caircocoders-ednalan"
  
UPLOAD_FOLDER = 'storage/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
  
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


@app.errorhandler(404)
def not_found(e):
    return app.send_static_file('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        resp = jsonify({
            "message": 'No file part in the request',
            "status": 'failed'
        })
        resp.status_code = 400
        return resp
    
    files = request.files.getlist('files[]')
      
    errors = {}
    success = False
    results = []
    for file in files:      
        if file and allowed_file(file.filename):
            results = predict(file)
            success = True
        else:
            resp = jsonify(
            {
                "message": 'File type is not allowed',
                "status": 'failed',
                "error": files
            })
            return resp
         
    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        errors['status'] = 'failed'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        resp = jsonify(
        {
            "message": 'Files successfully uploaded',
            "status": 'successs',
            "predictions": results
        })

        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
 
if __name__ == '__main__':
    app.run(debug=True)