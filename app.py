import json

from flask import Flask, abort, jsonify, render_template, request, Response

# from tools.JSONDisplay import actualDisplay
from layout_difference import MIPCompare
from layout_engine import ElementaryPlacement
from tools.JSONLoader import Layout

from optimizer import layout_difference, classes

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

    print("Comparing", first_layout.id, " ( n =", first_layout.n, ") <>",
          second_layout.id, "( n =", str(second_layout.n), ")")

    result = MIPCompare.solve(first_layout, second_layout)

    first_layout, second_layout = map(classes.Layout, json.loads(request.data))

    print("Comparing", first_layout.id, " ( n =", first_layout.n, ") <>",
          second_layout.id, "( n =", str(second_layout.n), ")")

    result2 = layout_difference.solve(first_layout, second_layout)

    return jsonify([result, result2])

@app.route('/api/v1.0/optimize-layout/', methods=['POST'])
def optimize_layout() -> Response:
    # TODO: consider adding checks for security
    return jsonify(ElementaryPlacement.solve(Layout(json.loads(request.data))))

@app.route('/api/v1.0/apply-template/', methods=['POST'])
def apply_template() -> Response:
    # TODO: consider adding checks for security
    request_props = json.loads(request.data)

    return jsonify(ElementaryPlacement.solve(Layout(request_props['layout']), Layout(request_props['template']), request_props['results']))

