from dataclasses import dataclass, field

@dataclass
class Direction:
    yaw: float
    pitch: float
    yaw_threshold: int = 20
    pitch_threshold: int = 20
    value: list = field(init=False, default_factory=lambda: [0, 0])

    def __post_init__(self):
        self._calculate_direction()

    def _calculate_direction(self) -> None:
        if self.yaw > self.yaw_threshold:
            self.value[0] = 1
        elif self.yaw < -self.yaw_threshold:
            self.value[0] = -1

        if self.pitch > self.pitch_threshold:
            self.value[1] = 1 # Down
        elif self.pitch < -self.pitch_threshold:
            self.value[1] = -1 # Up

    @property
    def x(self) -> int:
        return self.value[0]

    @property
    def y(self) -> int:
        return self.value[1]
    
    def __str__(self) -> str:
        horizontal_map = {-1: "Left", 0: "Straight", 1: "Right"}
        vertical_map = {-1: "Up", 0: "Straight", 1: "Down"}

        h_dir = horizontal_map.get(self.x, "Invalid")
        v_dir = vertical_map.get(self.y, "Invalid")

        if h_dir == "Straight" and v_dir == "Straight":
            return "Straight"
        
        return f"{h_dir if h_dir != 'Straight' else ''} {v_dir if v_dir != 'Straight' else ''}".strip().lower()

