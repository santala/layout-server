
class Layout:
    def __init__(self, props: dict):
        self.canvasWidth = None
        self.canvasHeight = None
        self.NumOfSolutions = None
        self.elements = []
        self.n = None

        props = props.get("layouts")[0] # TODO: edit the format

        self.canvasWidth = props.get('canvasWidth')
        self.canvasHeight = props.get('canvasHeight')
        self.NumOfSolutions = props.get('NumOfSolutions')

        self.elements = [
            Element(element_props) for element_props in props.get('elements')
        ]

        self.n = len(self.elements)


class Element:
    def __init__(self, props: dict):

        self.id = props.get('id')
        self.X = props.get('x')
        self.Y = props.get('y')
        self.width = props.get('width')
        self.height = props.get('height')
        self.minWidth = props.get('minWidth')
        self.minHeight = props.get('minHeight')
        self.maxWidth = props.get('maxWidth')
        self.maxHeight = props.get('maxHeight')
        self.horizontalPreference = props.get('horizontalPreference')
        self.verticalPreference = props.get('verticalPreference')
        self.aspectRatio = props.get('aspectRatio')
        self.elementType = props.get('type')

        if self.width is not None and self.width >= 0:
            self.minWidth = self.width
            self.maxWidth = self.width
        if self.height is not None and self.height >= 0:
            self.minHeight = self.height
            self.maxHeight = self.height

