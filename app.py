from flask import Flask, render_template, request
import model as m


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict",methods = ["POST"])
def predict():
    if request.method == "POST":
        user_text = request.form['user_text']
        model_pred = m.predict_sentiment(user_text)

    return render_template("index.html",
    text='Texto: {}'.format(user_text),
    pred='Sentimento: {}'.format(model_pred)
    )

"""
@app.route("/sub", methods = ['POST'])
def submit():
    # HTML -> .py
    if request.method == 'POST':
        texto = request.form["user_text"]
    
    #py. -> HTML
    return render_template("sub.html",n=texto)
"""

if __name__ == "__main__":
    app.run(debug=True)