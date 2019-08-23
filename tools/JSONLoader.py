
class Layout:
    def __init__(self, props: dict):
        props = props.get("layouts")[0]  # TODO: edit the format

        self.id = str(props.get('id'))
        self.canvas_width = props.get('canvasWidth', None)
        self.canvas_height = props.get('canvasHeight', None)
        self.solution_count = props.get('NumOfSolutions', None)

        self.elements = [
            Element(element_props) for element_props in props.get('elements', [])
        ]

        self.n = len(self.elements)

        self.x_sum = 0
        self.y_sum = 0
        self.w_sum = 0
        self.h_sum = 0
        self.area_sum = 0


class Element:
    def __init__(self, props: dict):

        self.id = str(props.get('id'))
        self.x = props.get('x')
        self.y = props.get('y')
        self.width = props.get('width', None)
        self.height = props.get('height', None)
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

