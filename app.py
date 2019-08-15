import json
from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

from tools.JSONDisplay import actualDisplay
from solver.PrepareParameters import prepare
from solver.MIPCompare import solve
from model import Element
from model import Layout

app = Flask(__name__)

@app.route('/')
def hello_world():
    try:
        message = request.args['message']
    except:
        message = 'Hello, World!'
    return render_template('test.html', message=message)

@app.route('/api/v1.0/', methods=['POST'])
def upload():

    # TODO: consider improving error reporting

    # Check existence of file parameters
    if 'layoutA' not in request.files or 'layoutB' not in request.files:
        return abort(400) # Bad request

    # Check existence of files
    layout_file_a = request.files['layoutA']
    layout_file_b = request.files['layoutB']

    print(layout_file_a, layout_file_b)

    if layout_file_a.filename == '' or layout_file_b.filename == '':
        return abort(418) # Either file was not found

    layout_a = json.load(layout_file_a.stream)
    layout_b = json.load(layout_file_b.stream)

    layout_file_a.close()
    layout_file_b.close()

    diff = findDifference(layout_a, layout_b)

    return jsonify(diff)


def findDifference(layout_a: dict, layout_b: dict, display_result=False) -> int:
    print("Comparing ", layout_a['layouts'][0]['id'], "<>", layout_b['layouts'][0]['id'])
    first_layout = convert_json(layout_a)
    second_layout = convert_json(layout_b)
    PenaltyAssignment = prepare(first_layout, second_layout)
    if display_result:
        actualDisplay(first_layout, "FirstLayout")
        actualDisplay(second_layout, "SecondLayout")
    return solve(first_layout, second_layout, PenaltyAssignment)


def convert_json(json_dict) -> Layout:
    data = Layout()
    JSONdata = json_dict.get("layouts")[0]
    data.canvasWidth = JSONdata.get('canvasWidth')
    data.canvasHeight = JSONdata.get('canvasHeight')
    data.id = JSONdata.get('id')

    JSONelements = JSONdata.get('elements')
    data.N = len(JSONelements)

    for JSONelement in JSONelements:
        element = Element()
        element.id = JSONelement.get('id')
        element.X = JSONelement.get('x')
        element.Y = JSONelement.get('y')
        element.width = JSONelement.get('width')
        element.height = JSONelement.get('height')
        data.elements.append(element)

    return data