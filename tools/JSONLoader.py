
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

        # The following are for the layout difference algorithm

        self.x_sum = sum([element.x for element in self.elements])
        self.y_sum = sum([element.y for element in self.elements])
        self.w_sum = sum([element.width for element in self.elements])
        self.h_sum = sum([element.height for element in self.elements])
        self.area_sum = sum([element.area for element in self.elements])

        # EXPL: Penalty of being skipped is the relative size of the element
        for element in self.elements:
            element.PenaltyIfSkipped = element.area / self.area_sum


class Element:
    def __init__(self, props: dict):

        self.id = str(props.get('id'))
        self.x = props.get('x')
        self.y = props.get('y')
        self.width = props.get('width', None)
        self.height = props.get('height', None)
        self.area = self.width * self.height if self.width is not None and self.height is not None else None
        self.minWidth = props.get('minWidth')
        self.minHeight = props.get('minHeight')
        self.maxWidth = props.get('maxWidth')
        self.maxHeight = props.get('maxHeight')
        self.horizontalPreference = props.get('horizontalPreference')
        self.verticalPreference = props.get('verticalPreference')
        self.aspectRatio = props.get('aspectRatio')
        self.elementType = props.get('type')

        self.PenaltyIfSkipped = None

        if self.width is not None and self.width >= 0:
            self.minWidth = self.width
            self.maxWidth = self.width
        if self.height is not None and self.height >= 0:
            self.minHeight = self.height
            self.maxHeight = self.height

