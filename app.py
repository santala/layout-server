import csv, io
from flask import Flask, abort, redirect, render_template, request, url_for

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

    # TODO: consider adding CSRF protection (with e.g. Flask-WTF) to limit the usage only to the front end

    message = ""
    if 'data_set' not in request.files:
        return abort(400) # Bad request

    message += 'File part found.\n'

    file = request.files['data_set']

    if file.filename == '':
        return abort(418) # File wasn’t selected

    message += 'File found!\n'

    with file.stream as f:
        reader = csv.reader(iter(f.readline, ''))

        print(reader)

        for row in reader:
            message += ', '.join(row) + '\n'

    #file.stream.close()

    return redirect(url_for('hello_world', message=message))