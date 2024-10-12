from flask import Flask, render_template, request
import google.generativeai as genai
import os
import numpy as np
import textblob as tb
import joblib
import yfinance as yf
from babel.numbers import format_currency
import pickle

model = genai.GenerativeModel("gemini-1.5-flash")
api = os.getenv("MAKERSUITE")
genai.configure(api_key="AIzaSyDgb0vUuFH_K46kU-0gmqkIcmIOdYQxgpE")

app = Flask(__name__)

# model2 = joblib.load('templates/bankruptcy_model.pkl')
with open('templates/bankruptcy_model.pkl', 'rb') as f:
    model2 = pickle.load(f)

def get_stock_data(q):
    # Get stock data using yfinance
    stock = yf.Ticker(q)
    stock_history = stock.history(period="5d")  # Fetch last 5 days of data
    
    # If not enough data, return None
    if stock_history.empty:
        return None

    # Extract previous and current prices
    previous_close = stock_history['Close'].iloc[-2]  # Previous day's close
    current_price = stock_history['Close'].iloc[-1]   # Latest closing price

    # Calculate dollar change and percent change
    dollar_change = current_price - previous_close
    percent_change = (dollar_change / previous_close) * 100

    # Formatted to SGD
    formatted_previous_close = format_currency(previous_close, 'SGD', locale='en_SG')
    formatted_current_price = format_currency(current_price, 'SGD', locale='en_SG')
    formatted_dollar_change = format_currency(dollar_change, 'SGD', locale='en_SG')
    formatted_percent_change = round(percent_change, 2)

    # Return the data
    return {
        'Previous Close': formatted_previous_close,
        'Current Price': formatted_current_price,
        'Dollar Change': formatted_dollar_change,
        'Percent Change': f"{formatted_percent_change}%"
    }

@app.route("/", methods = ["GET","POST"])
def index():
    return (render_template("index.html"))

@app.route("/in_class", methods = ["GET","POST"])
def in_class():
    return (render_template("in_class.html"))

@app.route("/individual_assignment", methods = ["GET","POST"])
def individual_assignment():
    return (render_template("individual_assignment.html"))

@app.route("/group_assignment", methods = ["GET","POST"])
def group_assignment():
    return (render_template("group_assignment.html"))

@app.route("/banks", methods = ["GET","POST"])
def banks():
    return (render_template("banks.html"))

@app.route("/reits", methods = ["GET","POST"])
def reits():
    return (render_template("reits.html"))

@app.route("/glcs", methods = ["GET","POST"])
def glcs():
    return (render_template("glcs.html"))

@app.route("/shares", methods = ["GET","POST"])
def shares():
    # Get the stock symbol from the form
    q = request.form['q']

    # Fetch stock data
    if q == "DBS":
        r = get_stock_data("D05.SI")
    elif q == "OCBC":
        r = get_stock_data("O39.SI")
    elif q == "UOB":
        r = get_stock_data("U11.SI")
    elif q == "Capitaland":
        r = get_stock_data("9CI.SI")
    elif q == "Frasers":
        r = get_stock_data("TQ5.SI")
    elif q == "Mapletree":
        r = get_stock_data("ME8U.SI")
    elif q == "Capitaland":
        r = get_stock_data("9CI.SI")
    elif q == "Frasers":
        r = get_stock_data("TQ5.SI")
    elif q == "Mapletree":
        r = get_stock_data("ME8U.SI")
    elif q == "Sembcorp Industries":
        r = get_stock_data("U96.SI")
    elif q == "Keppel Ltd":
        r = get_stock_data("BN4.SI")
    elif q == "Singapore Technologies":
        r = get_stock_data("S63.SI")
    else:
        r = get_stock_data(q)

    # If no data, handle the case
    if r is None:
        return render_template('shares.html', error=f"Could not fetch data for {q}. Please check the symbol.", r=None)
    
    # Render the page with the stock data
    return render_template('shares.html', r=r)

@app.route("/prediction_DBS", methods = ["GET","POST"])
def prediction_DBS():
    return (render_template("prediction_DBS.html"))

@app.route("/prediction_result_DBS", methods = ["GET","POST"])
def prediction_result_DBS():
    q = float(request.form.get("q"))
    r = (-50.6 * q) + 90.2
    return (render_template("prediction_result_DBS.html",r=r))

@app.route("/predict_Creditability", methods = ["GET","POST"])
def predict_Creditability():
    return (render_template("predict_Creditability.html"))

@app.route("/predict_Creditability_result", methods = ["GET","POST"])
def predict_Creditability_result():
    q = float(request.form.get("q"))
    r = (-0.00014068 * q) + 1.35314472
    r = np.where(r>=0.5, "CREDITABLE", "NOT CREDITABLE")
    return (render_template("predict_Creditability_result.html",r=r))

@app.route("/sentiment_analysis", methods = ["GET","POST"])
def sentiment_analysis():
    return (render_template("sentiment_analysis.html"))

@app.route("/sentiment_analysis_result", methods = ["GET","POST"])
def sentiment_analysis_result():
    q = request.form.get("q")
    r = tb.TextBlob(q).sentiment
    return (render_template("sentiment_analysis_result.html",r=r))

@app.route("/faq", methods = ["GET","POST"])
def faq():
    return (render_template("faq.html"))

@app.route("/q1", methods = ["GET","POST"])
def q1():
    r = model.generate_content("How should I diversify my investment portfolio?")
    return (render_template("q1_reply.html",r=r))

@app.route("/q2", methods = ["GET","POST"])
def q2():
    q = request.form.get("q")
    r = model.generate_content(q)
    return (render_template("q2_reply.html",r=r))

@app.route('/predict', methods=['POST'])
def predict():
    # Step 4: Get data from the form
    if request.method == 'POST':
        # Retrieve form data
        working_capital = float(request.form['working_capital'])
        retained_earnings = float(request.form['retained_earnings'])
        EBIT = float(request.form['EBIT'])
        total_liabilities = float(request.form['total_liabilities'])
        sales = float(request.form['sales'])
        equity = float(request.form['equity'])
        
        # Step 5: Make prediction using the Random Forest model
        data = np.array([[working_capital, retained_earnings, EBIT, total_liabilities, sales, equity]])
        prediction = model2.predict(data)
        
        # Step 6: Return the prediction result
        if prediction == 0:
            return render_template('individual_assignment.html', prediction_text='The company is NOT likely to go bankrupt.')
        else:
            return render_template('individual_assignment.html', prediction_text='The company is likely to go bankrupt.')
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run()
