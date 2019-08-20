import json

class ResultInstance():

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)

class ElementLevelResult():
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)

def SaveToJSon(N, CanvasSize_W, CanvasSize_H, Lval, Tval, Wval, Hval, solNo, data, objValue):
    layouts = dict()
    layouts['layouts'] = []
    thislayout = dict()
    thislayout['objectiveValue'] = objValue
    thislayout['canvasWidth'] = CanvasSize_W
    thislayout['canvasHeight'] = CanvasSize_H
    thislayout['id'] = solNo
    thislayout['elements'] = []
    for elementNo in range(N):
        content = dict()
        content['x'] = Lval[elementNo]
        content['y'] = Tval[elementNo]
        content['width'] = Wval[elementNo]
        content['height'] = Hval[elementNo]
        content['id'] = data.elements[elementNo].id
        thislayout['elements'].append(content)
    layouts['layouts'].append(thislayout)
    with open("output/"+(str(solNo)+".json"), "w") as write_file:
        json.dump(layouts, write_file)
        write_file.close()
