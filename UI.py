from flask import Flask, request,redirect, url_for, render_template
from werkzeug import secure_filename
app = Flask(__name__, static_folder='static/search/', static_url_path='')
import json
import sys
import os
import logging

import coffeescript

from PDFParser import PDFParser

UPLOAD_FOLDER = '/tmp/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
filePath = ""
htmlContent = None

@app.route('/', methods = ['GET','POST'])

def root():
    if request.method == 'POST':

        file = request.files['file']

        filename = secure_filename(file.filename)

        if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'])):
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER']))

        destination = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        app.PDFParser.convert_html(destination)

        return '', 204

    return app.send_static_file('index.html')



@app.route('/get-pdf', methods=['POST'])
def get_pdf():

    return app.PDFParser.readPDF(request.get_json())


    
def compile_assets():
    static = app.static_folder

    with open("{}/javascript/index.js".format(static), 'w') as f:
        infile = "{}/coffeescript/index.coffee".format(static)
        js = coffeescript.compile_file(infile)
        f.write(js)

def server():
    print('Compiling assets...')
    compile_assets()

    app.PDFParser = PDFParser()

    return app

if __name__ == '__main__':
    server().run(debug=True)