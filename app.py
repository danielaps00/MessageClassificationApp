from flask import Flask, render_template, request
import smtplib
from smtplib import *

import pandas as pd 
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = Flask(__name__)

data = pd.read_csv('data_teks.csv', delimiter=';', encoding='ISO-8859-1')

# stemmer 
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

# stopword
stop_factory = StopWordRemoverFactory()
stoper = stop_factory.create_stop_word_remover() 

def case_fold(teks):
  return teks.casefold() 

def tanda_baca(teks):
  return re.sub('\[[^]]*\]', '', teks)

def stem(teks):
  return stemmer.stem(teks)

def stop(teks):
  teks = stoper.remove(teks)
  return teks

def teks_bersih(teks):
  teks = case_fold(teks)
  teks = tanda_baca(teks)
  teks = stem(teks)
  teks = stop(teks)
  return teks
  
data['teks'] = data['teks'].apply(teks_bersih)

text = data['teks'].values.tolist()
label = data['kategori'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
pembobot = TfidfVectorizer()
bobot_data_latih = pembobot.fit_transform(X_train)

from sklearn.naive_bayes import MultinomialNB
model_latih = MultinomialNB().fit(bobot_data_latih, y_train)

bobot_data_uji = pembobot.transform(X_test)
pred = model_latih.predict(bobot_data_uji)

penerima1 = '2017103468@student.kalbis.ac.id'
penerima2 = 'gracemarket9@gmail.com'

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/main', methods=['POST'])
def main():
    global ser, email
    try:
      ser = smtplib.SMTP('smtp.gmail.com', 587)
      ser.starttls()

      email = request.form['email']
      password = request.form['pass']

      ser.login(email, password)
      return render_template('alert/login/login_sukses.html')
    except:
      return render_template('alert/login/email_password_salah.html')  

@app.route('/logout')
def logout():
    try:
      ser.quit()
      return render_template('alert/login/logout_sukses.html')  
    except:
      return render_template('login.html') 

@app.route('/klasifikasi', methods=['POST'])
def klasifikasi(model=model_latih, pembobot=pembobot):
    judul = request.form['subjek']
    teks = request.form['pesan']
    
    s = teks_bersih(teks)
    bobot_data_kasus = pembobot.transform([s])
    pred = model.predict(bobot_data_kasus)
  
    if pred == 'Academic Operation':
      Message = 'Subject: {}\n\n{}'.format(judul, teks)
      ser.sendmail(email, penerima1, Message)
      return render_template('alert/main/Academic.html')
    elif pred == 'Finance':
      Message = 'Subject: {}\n\n{}'.format(judul, teks)
      ser.sendmail(email, penerima2, Message)
      return render_template('alert/main/Finance.html')

if __name__ == "__main__":
    app.run(debug=True)