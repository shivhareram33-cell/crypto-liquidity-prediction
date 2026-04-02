from flask import Flask, request , render_template
import joblib
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/submit" , methods=['POST'])
def submit():

    price = float(request.form['price'])
    houre_1 = float(request.form['1h'])
    houre_24 = float(request.form['24h'])
    day_7 = float(request.form['7d'])
    volume_24h = float(request.form['24h_volume'])
    mkt_cap = float(request.form['mkt_cap'])

    input = np.array([[price , houre_1 , houre_24 , day_7 , volume_24h , mkt_cap]])
    model = joblib.load('./notebook/final_model.pkl')
    prediction = model.predict(input)

    return render_template('index.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)


