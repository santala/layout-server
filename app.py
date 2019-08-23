import json

from flask import Flask, abort, jsonify, render_template, request, Response

# from tools.JSONDisplay import actualDisplay
from layout_difference import MIPCompare
from layout_engine import ElementaryPlacement
from tools.JSONLoader import Layout

app = Flask(__name__)

# TODO: configure HTTPS support
# https://blog.miguelgrinberg.com/post/running-your-flask-application-over-https

@app.route('/')
def hello_world() -> Response:
    try:
        message = request.args['message']
    except:
        message = 'Hello, World!'
    return render_template('test.html', message=message)


@app.route('/api/v1.0/layout-difference/', methods=['POST'])
def upload() -> Response:
    # TODO: consider adding checks for security
    first_layout, second_layout = map(Layout, json.loads(request.data))

    print("Comparing ", first_layout.id, " ( n =", first_layout.n, ") <>",
          second_layout.id, "( n =", str(second_layout.n), ")")

    return jsonify(MIPCompare.solve(first_layout, second_layout))

@app.route('/api/v1.0/optimize-layout/', methods=['POST'])
def optimize_layout() -> Response:
    # TODO: consider adding checks for security
    return jsonify(ElementaryPlacement.solve(Layout(json.loads(request.data))))

