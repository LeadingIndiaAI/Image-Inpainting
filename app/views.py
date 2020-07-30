from app import app

from flask import render_template, request, redirect

import os

#CHANGE THE FOLLOWING PATH !!
app.config['IMAGE_UPLOADS'] = '/Users/Bharat/Desktop/Flask Webapp/uploads'

@app.route('/', methods = ['GET', 'POST'])
def index():

    if request.method == 'POST':
        
        if request.files:

            image = request.files['image']

            # print(image)
            image.save(os.path.join(app.config['IMAGE_UPLOADS'], image.filename))

            print('Image uploaded')

            return redirect(request.url)
             
    return render_template('public/index.html')
