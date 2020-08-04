from webapp import app

from flask import render_template, request, redirect

import os

import time

import webapp.modelRunner.runner

from shutil import copyfile

# Create two constant. They direct to the app root folder and uploads folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__)) #to get the current working directory
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/uploads')
STATIC_FOLDER = os.path.join(APP_ROOT, 'static/')

app.config['IMAGE_UPLOADS'] = UPLOAD_FOLDER
app.config['CACHED_OUTPUT'] = STATIC_FOLDER

@app.route('/')
def home_page():
    return render_template('public/index.html')

@app.route('/upload', methods = ['GET', 'POST'])
def index():

    showoutput = False
    if request.method == 'POST':
        
        if request.files:

            # if os.path.exists(UPLOAD_FOLDER+'/img/input.png'):
            #     os.remove(UPLOAD_FOLDER+'/img/input.png')
            # if os.path.exists(UPLOAD_FOLDER+'/mask/input_mask.png'):
            #     os.remove(UPLOAD_FOLDER+'/mask/input_mask.png')

            # image = request.files['image']
            image = request.files.getlist('image')
            
            # print(image)
            # print(image.filename)
            zfilename = image[0].filename
            zfilename_mask = image[1].filename 
            
            # print(image[0].filename)
            # print(image[1].filename)
            print()
            print(zfilename_mask)
            print()
            # for single file
            # image.save(os.path.join(app.config['IMAGE_UPLOADS'], image.filename))

            image[0].save(os.path.join(app.config['IMAGE_UPLOADS'], 'img', zfilename))
            image[1].save(os.path.join(app.config['IMAGE_UPLOADS'], 'mask', zfilename_mask))

            #print('Image uploaded')

            option = request.form['options']
            print(option)
            ###Processing Code Here For The GAN Module
            #import script here for the GAN Module, do the processing and save it in a folder app/processed
            ##with the same filename + '_output'
            ##provide the image source to the next render_template where the image will be displayed
            # time.sleep(5)   #simulate processing
            # return redirect('/view_output')
            input_path = os.path.join(app.config['IMAGE_UPLOADS'], 'img', zfilename)  #input image path
            mask_path = os.path.join(app.config['IMAGE_UPLOADS'], 'mask', zfilename_mask)  #output image path
            showoutput = True 
            #print(input_path)
            #print(output_path)

            ####
            webapp.modelRunner.runner.imginp(input_path, mask_path)
            # time.sleep(2)

            outputfilename = 'image.jpg'

            #CHANGE HERE
            output_to_move = os.path.join(app.config['CACHED_OUTPUT'], 'image.jpg')
            moved_file = os.path.join(app.config['IMAGE_UPLOADS'], 'output', zfilename.split('.')[0] + '.' + outputfilename.split('.')[1])
            # os.system('cp '+output_to_move+' '+ moved_file)
            # print(output_to_move)
            # print(moved_file)
            print('cp '+output_to_move+' '+ moved_file)
            copyfile(output_to_move, moved_file)
            # url1 = "{{ url_for('static', filename='uploads/img/abc.png') }}"
            return render_template('/public/view-output.html', option = option, zfilename = zfilename, zfilename_mask = zfilename_mask, outputfilename = outputfilename)
            # return render_template('/public/view-output.html', option = option, zfilename = zfilename, zfilename_mask = zfilename_mask, check = check)

    return render_template('public/upload_image.html')


@app.route('/view_output', methods=['GET', 'POST'])
def view_output():
    return render_template('public/view-output.html')