
class DataInstance:
    def __init__(self):
        self.canvasWidth = None
        self.canvasHeight = None
        self.NumOfSolutions = None
        self.elements = []
        self.N = None


class Element:
    def __init__(self):
        self.width  = None
        self.height = None
        self.minWidth  = None
        self.minHeight = None
        self.maxWidth  = None
        self.maxHeight = None
        self.aspectRatio = None
        self.horizontalPreference = None
        self.verticalPreference = None
        self.elementType = None
        self.X = None
        self.Y = None
        self.id = None


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
