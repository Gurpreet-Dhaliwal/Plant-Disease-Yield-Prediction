# -*- coding: utf-8 -*-
import string

from scripts import tabledef
from scripts import forms
from scripts import helpers
from flask import Flask, redirect, url_for, render_template, request, session
import json
import sys
import os
from werkzeug.utils import secure_filename
from scripts import final_test
from scripts import description
import pandas as pd
import operator

UPLOAD_FOLDER = 'D:\\capstone project\\Flaskex-master\\Flaskex-master\\static\\uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
pic_path = ""
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = os.urandom(12)  # Generic key for dev purposes only
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Heroku
#from flask_heroku import Heroku
#heroku = Heroku(app)

# ======== Routing =========================================================== #
# -------- Login ------------------------------------------------------------- #
@app.route('/', methods=['GET', 'POST'])
def login():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = request.form['password']
            if form.validate():
                if helpers.credentials_valid(username, password):
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Login successful'})
                return json.dumps({'status': 'Invalid user/pass'})
            return json.dumps({'status': 'Both fields required'})
        return render_template('login.html', form=form)
    user = helpers.get_user()
    return render_template('home.html', user=user)


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))


# -------- Signup ---------------------------------------------------------- #
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = helpers.hash_password(request.form['password'])
            email = request.form['email']
            if form.validate():
                if not helpers.username_taken(username):
                    helpers.add_user(username, password, email)
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Signup successful'})
                return json.dumps({'status': 'Username taken'})
            return json.dumps({'status': 'User/Pass required'})
        return render_template('login.html', form=form)
    return redirect(url_for('login'))


# -------- Settings ---------------------------------------------------------- #
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if session.get('logged_in'):
        if request.method == 'POST':
            password = request.form['password']
            if password != "":
                password = helpers.hash_password(password)
            email = request.form['email']
            helpers.change_user(password=password, email=email)
            return json.dumps({'status': 'Saved'})
        user = helpers.get_user()
        return render_template('settings.html', user=user)
    return redirect(url_for('login'))

# -------- disease ---------------------------------------------------------- #
@app.route('/disease', methods=['GET', 'POST'])
def disease():
    #if session.get('logged_in'):
    if(True):
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                print('No file part')
                return redirect(request.url)
            file = request.files['file']
            print(file.filename)
            # if user does not select file, browser also
            # submit a empty part without filename
            if file.filename == '':
                print('No selected file')
                return redirect(request.url)
            else:
                filename = secure_filename(file.filename)
                filename1 = "static/uploads/"+  filename
                pic_path = filename1
                complete_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                print(pic_path)
                print(complete_path)
                #predict the category of the disease
                category = final_test.perdict(complete_path)
                #category = "c_7"
                d1 = {'c_0': 'Apple Scab, Venturia inaequalis', 'c_1': 'Apple Black Rot,Botryosphaeria obtusa',
                      'c_10': 'Apple Cedar Rust, Gymnosporangium juniperi-virginianae',
                      'c_11': 'Cherry Powdery Mildew, Podosphaera spp',
                      'c_12': 'Corn Gray Leaf Spot, Cercospora zeae-maydis',
                      'c_13': 'Corn Common Rust, Puccinia sorghi',
                      'c_14': 'Corn Northern Leaf Blight, Exserohilum turcicum',
                      'c_15': 'Grape Black Rot, Guignardia bidwellii',
                      'c_16': 'Grape Black Measles (Esca), Phaeomoniella aleophilum, Phaeomoniella chlamydospora',
                      'c_17': 'Grape Leaf Blight, Pseudocercospora vitis',
                      'c_18': 'Orange Huanglongbing (Citrus Greening),Candidatus Liberibacter spp',
                      'c_19': 'Peach Bacterial Spot, Xanthomonas campestris',
                      'c_2': 'Bell Pepper Bacterial Spot, Xanthomonas campestris',
                      'c_20': 'Potato Early Blight, Alternaria solani',
                      'c_21': 'Potato Late Blight, Phytophthora infestans ',
                      'c_23': 'Squash Powdery Mildew, Erysiphe cichoracearum, Sphaerotheca fuliginea',
                      'c_24': 'Strawberry Leaf Scorch, Diplocarpon earlianum',
                      'c_25': 'Tomato Bacterial Spot, Xanthomonas campestris pv vesicatoria',
                      'c_26': 'Tomato Early Blight, Alternaria solani',
                      'c_27': 'Tomato Late Blight, Phytophthora infestans', 'c_28': 'Tomato Leaf Mold, Fulvia fulva',
                      'c_29': 'Tomato Septoria Leaf Spot, Septoria lycopersici',
                      'c_3': 'Tomato Two Spotted Spider Mite, Tetranychus urticae',
                      'c_30': 'Tomato Target Spot, Corynespora cassiicola', 'c_31': 'Tomato Mosaic Virus',
                      'c_32': 'Tomato Yellow Leaf Curl Virus', 'c_33': 'Apple Scab, Venturia inaequalis',
                      'c_34': 'Corn Gray Leaf Spot, Cercospora zeae-maydis',
                      'c_35': 'Squash Powdery Mildew, Erysiphe cichoracearum, Sphaerotheca fuliginea',
                      'c_36': 'Strawberry Leaf Scorch, Diplocarpon earlianum',
                      'c_37': 'Tomato Two Spotted Spider Mite, Tetranychus urticae',
                      'c_4': 'Potato Early Blight, Alternaria solani', 'c_5': 'Tomato Mosaic Virus',
                      'c_6': 'Corn Gray Leaf Spot, Cercospora zeae-maydis',
                      'c_7': 'Peach Bacterial Spot, Xanthomonas campestris',
                      'c_8': 'Bell Pepper Bacterial Spot, Xanthomonas campestris',
                      'c_9': 'Tomato Bacterial Spot, Xanthomonas campestris pv vesicatoria'}

                name = category
                print(name)
                plant_name = name
                treatment_technique = description.description(d1[name])
                print(treatment_technique)
                d = {}
                d[category] = d1[name]
                return render_template('disease_output.html', plant_name=plant_name, treatment_technique=treatment_technique, plant_img=pic_path, dictionary=d)
        return render_template('upload_image.html')
    return redirect(url_for('login'))


# -------- Yield prediction  ---------------------------------------------------------- #
@app.route('/yield', methods=['GET', 'POST'])
def yield_predict():
    #if session.get('logged_in'):
    if(True):
        if request.method == 'POST':
            state1 = request.form['state']
            print("inside yield predict")

            if state1 != "":
                name = 'output/final.csv'
                state = string.upper(state1)
                print(state)
                crops = ['Yield_rice', 'Yield_sugarcane','Yield_cotton', 'Yield_oilseeds', 'Yield_pulses']
                cropindex_dict = {'Yield_rice': 2, 'Yield_wheat': 3, 'Yield_sugarcane': 4, 'Yield_cotton': 5,
                                  'Yield_oilseeds': 6, 'Yield_pulses': 7}
                price = [1700, 5150, 3300, 6000]
                attributes = ['rain_Jan', 'rain_Feb', 'rain_Mar', 'rain_Apr', 'rain_May', 'rain_Jun', 'rain_Jul',
                              'rain_Aug', 'rain_Sep', 'rain_Oct', 'rain_Nov', 'rain_Dec', 'cc_Jan', 'cc_Feb', 'cc_Mar',
                              'cc_Apr', 'cc_May', 'cc_Jun', 'cc_Jul', 'cc_Aug', 'cc_Sep', 'cc_Oct', 'cc_Nov', 'cc_Dec',
                              'max_temp_Jan', 'max_temp_Feb', 'max_temp_Mar', 'max_temp_Apr', 'max_temp_May',
                              'max_temp_Jun', 'max_temp_Jul', 'max_temp_Aug', 'max_temp_Sep', 'max_temp_Oct',
                              'max_temp_Nov', 'max_temp_Dec', 'min_temp_Jan', 'min_temp_Feb', 'min_temp_Mar',
                              'min_temp_Apr', 'min_temp_May', 'min_temp_Jun', 'min_temp_Jul', 'min_temp_Aug',
                              'min_temp_Sep', 'min_temp_Oct', 'min_temp_Nov', 'min_temp_Dec']
                states = ['HIMACHAL PRADESH', 'BIHAR', 'KERALA', 'WEST BENGAL', 'GUJARAT', 'PUNJAB', 'UTTAR PRADESH',
                          'TAMIL NADU']
                ind = [2,4, 5, 6, 7]
                # Importing the dataset
                list = []

                dataset = pd.read_csv(name)
                df = pd.DataFrame(dataset)
                dataset = df.loc[df['State'] == state]
                y = dataset.iloc[:, ind].values
                dict_price = {}

                dict_yield = {}
                for i in range(0, len(y[0])):
                    dict_yield[crops[i]] = y[0][i]
                sorted_x = sorted(dict_yield.items(), key=operator.itemgetter(1))
                print(sorted_x)
                print(state)
            return render_template('yield_predict_display.html', list= sorted_x)
        return render_template('yield_predict.html')
    return redirect(url_for('login'))


# ======== Main ============================================================== #
if __name__ == "__main__":
    app.run(debug=True, use_reloader=True, port=5000)
