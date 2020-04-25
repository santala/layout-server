import json, datetime, time
from flask import Flask, abort, jsonify, render_template, request, Response


from layout_engine import ElementaryPlacement
from tools import JSONLoader

from optimizer import classes, layout_difference, layout_quality, guidelines, model

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

@app.route('/api/v1.0/log-action/', methods=['GET'])
def log_action() -> Response:
    with open("action_log.txt", "a") as myfile:
        myfile.write(str(datetime.datetime.now())+"\n")
    return Response(status=200)

@app.route('/api/v1.0/optimize-layout/', methods=['POST'])
def optimize_layout() -> Response:
    # TODO: consider adding checks for security
    request_props = json.loads(request.data)


    time_out = int(request_props.get('timeOut', 30))
    snap_tolerance = float(request_props.get('snapTolerance', 8))
    gutter = float(request_props.get('gridGutter', 8))
    margin = float(request_props.get('gridMargin', 16))
    column_count = int(request_props.get('columnCount', 24))
    baseline = float(request_props.get('baseline', 8))

    if False:
        layout = classes.Layout(request_props['layout'])
        number_of_solutions = int(request_props.get('numberOfSolutions', 1))
        with open('./output/layouts/' + str(int(time.time_ns() / 1000)) + '.json', 'w', encoding='utf-8') as f:
            json.dump(request_props['layout'], f, ensure_ascii=False, indent=4)
    # Testing code
    #guidelines.solve(layout)
    print('Request Props', request_props)
    print('Solving…')
    if False:
        result = guidelines.solve(layout, number_of_solutions=number_of_solutions, time_out=time_out)
    else:
        result = model.solve(request_props['layout'],
                             time_out=time_out,
                             columns=column_count,
                             gutter=gutter,
                             margin=margin,
                             tolerance=snap_tolerance,
                             baseline=baseline
                             )

    return jsonify(result)


@app.route('/api/v1.0/apply-template/', methods=['POST'])
def apply_template() -> Response:
    # TODO: consider adding checks for security
    request_props = json.loads(request.data)

    return jsonify(ElementaryPlacement.solve(classes.Layout(request_props['layout']), classes.Layout(request_props['template']), request_props['results']))
