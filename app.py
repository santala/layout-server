import json

from flask import Flask, abort, jsonify, render_template, request, Response


from layout_engine import ElementaryPlacement
from tools import JSONLoader

from optimizer import classes, layout_difference, layout_quality

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
    first_layout, second_layout = map(JSONLoader.Layout, json.loads(request.data))

    first_layout, second_layout = map(classes.Layout, json.loads(request.data))

    print("Comparing", first_layout.id, " ( n =", first_layout.n, ") <>",
          second_layout.id, "( n =", str(second_layout.n), ")")

    result = layout_difference.solve(first_layout, second_layout)

    return jsonify(result)

@app.route('/api/v1.0/optimize-layout/', methods=['POST'])
def optimize_layout() -> Response:
    # TODO: consider adding checks for security
    request_props = json.loads(request.data)
    layout = classes.Layout(request_props['layout'])
    time_out = min(30, max(1, int(request_props.get('timeOut', 30))))
    number_of_solutions = min(10, max(1, int(request_props.get('numberOfSolutions', 1))))
    return jsonify(layout_quality.solve(layout, time_out=time_out, number_of_solutions=number_of_solutions))


@app.route('/api/v1.0/apply-template/', methods=['POST'])
def apply_template() -> Response:
    # TODO: consider adding checks for security
    request_props = json.loads(request.data)

    return jsonify(ElementaryPlacement.solve(classes.Layout(request_props['layout']), classes.Layout(request_props['template']), request_props['results']))
