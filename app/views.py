from app import app

from flask import render_template, request, redirect

import os

import time

# Create two constant. They direct to the app root folder and uploads folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__)) #to get the current working directory
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')

app.config['IMAGE_UPLOADS'] = UPLOAD_FOLDER

@app.route('/')
def home_page():
    return render_template('public/index.html')

@app.route('/upload', methods = ['GET', 'POST'])
def index():

    if request.method == 'POST':
        
        if request.files:

            image = request.files['image']

            # print(image)
            image.save(os.path.join(app.config['IMAGE_UPLOADS'], image.filename))

            print('Image uploaded')

            ###Processing Code Here For The GAN Module
            #import script here for the GAN Module, do the processing and save it in a folder app/processed
            ##with the same filename + '_output'
            ##provide the image source to the next render_template where the image will be displayed
            time.sleep(5)   #simulate processing
            return redirect('/view_output')
             
    return render_template('public/upload_image.html')


@app.route('/view_output', methods=['GET', 'POST'])
def view_output():
    return render_template('public/view-output.html')