import json

class DataInstance:
    canvasWidth = None
    canvasHeight = None
    NumOfSolutions = None
    elements = []
    N = None

class Element:
    width  = None
    height = None
    minWidth  = None
    minHeight = None
    maxWidth  = None
    maxHeight = None
    aspectRatio = None
    horizontalPreference = None
    verticalPreference = None
    elementType = None
    X = None
    Y = None
    id = None


def json_to_data_instance(json_dict) -> DataInstance:
    print(json_dict)
    data = DataInstance()
    JSONdata = json_dict.get("layouts")[0]
    data.canvasWidth = JSONdata.get('canvasWidth')
    data.canvasHeight = JSONdata.get('canvasHeight')
    data.NumOfSolutions = JSONdata.get('NumOfSolutions')
    JSONelements = JSONdata.get('elements')
    data.N = len(JSONelements)
    for JSONelement in JSONelements:
        element = Element()
        element.id = JSONelement.get('id')
        element.X = JSONelement.get('x')
        element.Y = JSONelement.get('y')
        element.width = JSONelement.get('width')
        element.height = JSONelement.get('height')
        element.minWidth = JSONelement.get('minWidth')
        element.minHeight = JSONelement.get('minHeight')
        element.maxWidth = JSONelement.get('maxWidth')
        element.maxHeight = JSONelement.get('maxHeight')
        element.horizontalPreference = JSONelement.get('horizontalPreference')
        element.verticalPreference = JSONelement.get('verticalPreference')
        element.aspectRatio = JSONelement.get('aspectRatio')
        element.elementType = JSONelement.get('type')


        if(element.width is not None and element.width >= 0):
            element.minWidth = element.width
            element.maxWidth = element.width
        if (element.height is not None and element.height >= 0):
            element.minHeight = element.height
            element.maxHeight = element.height

        data.elements.append(element)
    return data
