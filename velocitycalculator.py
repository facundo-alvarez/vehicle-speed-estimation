class VelocityCalculator():
    def __init__(self) -> None:
        self.ids = {}

    def calculate(self, tracker: tuple, fps: int) -> int:
        id = tracker[0]
        x = tracker[1]
        y = tracker[2]

        if id not in self.ids:
            self.ids[id] = [x, y, 0, 0]
        else:
            self.ids[id][3] += 1
            if self.ids[id][3] == fps // 2:  # Update every half second
                distance = abs(y - self.ids[id][1])
                speed = distance * 2 * 3.6
                self.ids[id][0] = x
                self.ids[id][1] = y
                if (self.ids[id][2] != 0):
                    self.ids[id][2] = int((speed + self.ids[id][2]) / 2)
                else:
                    self.ids[id][2] = int(speed)
                self.ids[id][3] = 0

        return self.ids[id][2]  # Return the existing speed without recalculating
