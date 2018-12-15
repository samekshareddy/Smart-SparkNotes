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
filePath = ""
htmlContent = None

@app.route('/', methods = ['GET','POST'])

def root():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
      
        f.save(os.path.join(UPLOAD_FOLDER, filename))

        filePath = os.path.join(UPLOAD_FOLDER, filename)

        app.PDFParser.convert_html(filePath)

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