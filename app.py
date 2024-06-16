import pickle as pkl
import numpy as np
from flask import Flask,render_template,request


model=pkl.load(open('model.pkl','rb'))
app = Flask(__name__, template_folder='templates')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
     age=request.form.get('age')
     sex=request.form.get('sex')
     cp=request.form.get('cp')
     trtps=request.form.get('trtps')
     chol=request.form.get('chol')
     fbs=request.form.get('fb')
     restecg=request.form.get('restecg')
     thalach=request.form.get('thalach')
     exang=request.form.get('exang')
     oldpeak=request.form.get('oldpeak')
     slope=request.form.get('slope')
     ca=request.form.get('ca')
     thal=request.form.get('thal')
     result = model.predict(np.array([age,sex,cp,trtps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]).reshape(1,13))
    
     return result

if __name__== '__main__':
    app.run(debug=True ,port=8002)

