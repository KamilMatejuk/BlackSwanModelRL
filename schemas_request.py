import re
from dataclasses import dataclass
 

@dataclass
class State:
    price: float
    volume: float
    rsi: float
    macd: float
    EMA_12: float
    EMA_26: float
    value_percent_in_account: float
    value_percent_in_assets: float


@dataclass
class ActionRequest:
    id: str
    state: State
    
    def validate(self):
        self.state = State(**self.state)


@dataclass
class LearnRequest:
    id: str
    state: State
    action: int
    next_state: State
    reward: float
    
    def validate(self):
        self.state = State(**self.state)
        self.next_state = State(**self.next_state)
