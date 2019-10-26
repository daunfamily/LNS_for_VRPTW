class Route:
    def __init__(self):
        self.r_list = []
        self.dis = 0

    def __copy__(self):
        new_route = Route()
        new_route.r_list = [i for i in self.r_list]
        new_route.dis = self.dis


