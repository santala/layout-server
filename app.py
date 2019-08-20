import json

from flask import Flask, abort, jsonify, render_template, request, Response

# from tools.JSONDisplay import actualDisplay
from layout_difference import Layout
from layout_difference.MIPCompare import solve
from layout_difference.PrepareParameters import build_layout_parameters
from layout_engine import ElementaryPlacement
from tools.JSONLoader import json_to_data_instance

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

    # TODO: consider improving error reporting (or not, for security reasons)

    # Check existence of file parameters
    if 'layoutA' not in request.files or 'layoutB' not in request.files:
        return abort(400) # Bad request

    # Check existence of files
    layout_file_a = request.files['layoutA']
    layout_file_b = request.files['layoutB']

    print(layout_file_a, layout_file_b)

    if layout_file_a.filename == '' or layout_file_b.filename == '':
        return abort(418) # File missing

    layout_a = json.load(layout_file_a.stream)
    layout_b = json.load(layout_file_b.stream)

    layout_file_a.close()
    layout_file_b.close()

    diff = compute_difference(layout_a, layout_b)

    return jsonify(diff)

@app.route('/api/v1.0/optimize-layout/', methods=['POST'])
def optimize_layout() -> Response:

    # TODO: consider improving error reporting (or not, for security reasons)

    layout = json.loads(request.data)
    '''    
    # Check existence of file parameters
    if 'layout' not in request.files:
        return abort(400) # Bad request

    # Check existence of files
    layout_file = request.files['layout']

    if layout_file.filename == '':
        return abort(418) # File missing

    layout = json.load(layout_file.stream)

    layout_file.close()
    '''
    optimized = ElementaryPlacement.solve(json_to_data_instance(layout))

    return jsonify(optimized)



def compute_difference(layout_a: dict, layout_b: dict, display_result=False) -> int:
    print("Comparing ", layout_a['layouts'][0]['id'], "<>", layout_b['layouts'][0]['id'])
    first_layout = Layout(layout_a.get("layouts")[0])
    second_layout = Layout(layout_b.get("layouts")[0])
    penalty_assignment = build_layout_parameters(first_layout, second_layout)
    '''
    if display_result:
        actualDisplay(first_layout, "FirstLayout")
        actualDisplay(second_layout, "SecondLayout")
    '''
    return solve(first_layout, second_layout, penalty_assignment)
