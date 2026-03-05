from flask import Flask, render_template, request, redirect, url_for, session, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "secret-key"  # Needed for session

# Load your trained brain tumor prediction model
model = load_model("brain_tumor_prediction_model.h5")
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dummy in-memory user store (for example only)
users = {}

@app.route('/')
def home():
    user_logged_in = 'user' in session
    return render_template('home.html', user_logged_in=user_logged_in)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in users:
            flash('User already exists')
        else:
            users[email] = password
            flash('Registered successfully')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    next_page = request.args.get('next')  # Capture where to go after login
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if users.get(email) == password:
            session['user'] = email
            flash('Login successful!')
            return redirect(url_for(next_page)) if next_page else redirect(url_for('home'))
        else:
            flash('Invalid credentials or user not registered')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out')
    return redirect(url_for('home'))

# 🔒 Require login for prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        flash('Please register if you are new, or login if you already have an account.')
        return redirect(url_for('login', next='predict'))

    prediction = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            pred = model.predict(img_array)
            prediction = class_names[np.argmax(pred)]
            image_path = filepath

    return render_template('index.html', prediction=prediction, image_path=image_path)

# 🔒 Require login for About
@app.route('/about')
def about():
    if 'user' not in session:
        flash('Please register if you are new, or login if you already have an account.')
        return redirect(url_for('login', next='about'))
    return render_template('about.html')

# 🔒 Require login for Contact
@app.route('/contact')
def contact():
    if 'user' not in session:
        flash('Please register if you are new, or login if you already have an account.')
        return redirect(url_for('login', next='contact'))
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)