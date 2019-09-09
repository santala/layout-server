
class Layout:
    def __init__(self, props: dict):
        # TODO: format validation
        props = props.get('layouts')[0]  # TODO: edit the format

        self.id = str(props.get('id'))
        self.canvas_width = props.get('canvasWidth', None)
        self.canvas_height = props.get('canvasHeight', None)
        self.solution_count = props.get('NumOfSolutions', None)

        self.elements = [
            Element(element_props) for element_props in props.get('elements', [])
        ]

        self.n = len(self.elements)

        # The following are for the layout difference algorithm
        # TODO: consider making this a separate method

        self.x_sum = sum([abs(element.x) for element in self.elements])
        self.y_sum = sum([abs(element.y) for element in self.elements])
        self.w_sum = sum([abs(element.width) for element in self.elements])
        self.h_sum = sum([abs(element.height) for element in self.elements])
        self.area_sum = sum([element.area for element in self.elements])

        # EXPL: Penalty of being skipped is the relative size of the element
        for element in self.elements:
            element.PenaltyIfSkipped = element.area / self.area_sum



class Element:
    def __init__(self, props: dict):


        self.id = str(props.get('id'))
        self.x = int(props.get('x'))
        self.y = int(props.get('y'))
        self.width = props.get('width', None)
        self.height = props.get('height', None)
        self.area = self.width * self.height \
            if self.width is not None and self.height is not None \
               and self.width > 0 and self.height > 0 \
            else None
        self.horizontalPreference = props.get('horizontalPreference')
        self.verticalPreference = props.get('verticalPreference')
        self.aspectRatio = props.get('aspectRatio', None)
        self.elementType = props.get('type')
        self.componentName = props.get('componentName', '?')

        print(self.elementType, self.componentName)

        self.constrainLeft = bool(props.get('constrainLeft', False))
        self.constrainRight = bool(props.get('constrainRight', False))
        self.constrainTop = bool(props.get('constrainTop', False))
        self.constrainBottom = bool(props.get('constrainBottom', False))
        self.constrainWidth = bool(props.get('constrainWidth', False))
        self.constrainHeight = bool(props.get('constrainHeight', False))

        self.isLocked = bool(props.get('isLocked', False))

        self.PenaltyIfSkipped = None

        # TODO: make grid size configurable

        if self.width is not None and self.width >= 0:
            self.minWidth = int(self.width / 8) * 8
            self.maxWidth = int(self.width / 8 + 1) * 8
        if self.height is not None and self.height >= 0:
            self.minHeight = int(self.height / 8) * 8
            self.maxHeight = int(self.height / 8 + 1) * 8


