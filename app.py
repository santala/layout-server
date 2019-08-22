import json

from flask import Flask, abort, jsonify, render_template, request, Response

# from tools.JSONDisplay import actualDisplay
import layout_difference
from layout_difference import MIPCompare
from layout_difference.PrepareParameters import build_layout_parameters
from layout_engine import ElementaryPlacement
from tools import JSONLoader

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
    first_layout, second_layout = map(layout_difference.Layout, json.loads(request.data))

    print("Comparing ", first_layout.id, " ( n =", first_layout.n, ") <>",
          second_layout.id, "( n =", str(second_layout.n), ")")

    penalty_assignment = build_layout_parameters(first_layout, second_layout)

    return jsonify(MIPCompare.solve(first_layout, second_layout, penalty_assignment))

@app.route('/api/v1.0/optimize-layout/', methods=['POST'])
def optimize_layout() -> Response:
    # TODO: consider adding checks for security
    return jsonify(ElementaryPlacement.solve(JSONLoader.Layout(json.loads(request.data))))

