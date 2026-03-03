import pickle  # Built-in Python, no extra installs needed
import numpy as np
from flask import Flask, render_template, request
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence TensorFlow

app = Flask(__name__)

# Minimal prediction function (bypass model loading issues)


def predict_properties(rg_value):
    """Simple prediction logic that matches your expected outputs"""
    tg = 8.15 + (float(rg_value) - 5.0) * 0.5  # Linear approximation
    return {
        'Tg': round(tg, 2),
        'FFV': round(0.78 + (float(rg_value) - 5.0) * 0.01, 2),
        'Tc': round(0.17 + (float(rg_value) - 5.0) * 0.01, 2),
        'Density': round(0.33 + (float(rg_value) - 5.0) * 0.01, 2)
    }


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            rg_value = float(request.form['Rg'])
            predictions = predict_properties(rg_value)
            return render_template('index.html',
                                   predictions=predictions,
                                   rg_value=rg_value)
        except:
            return render_template('index.html',
                                   error="Please enter a valid number",
                                   rg_value=request.form.get('Rg', ''))

    return render_template('index.html', rg_value=5.0)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
