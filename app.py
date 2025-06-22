from flask import Flask, request, jsonify, render_template
from backend import run_deforestation_pipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    result = run_deforestation_pipeline(
        lat_min=data['lat_min'],
        lat_max=data['lat_max'],
        lon_min=data['lon_min'],
        lon_max=data['lon_max'],
        start_year=data['start_year'],
        end_year=data['end_year']
    )

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

