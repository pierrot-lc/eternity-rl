import dataclasses

@dataclasses.dataclass
class MCTSConfig:
    c_puct: float
    gamma: float
    simulations: int
    childs: int
