import re
from dataclasses import dataclass


@dataclass
class URL:
    host: str
    port: int
    slug: str
    
    def validate(self, name: str):
        assert self.host == "localhost" or \
            re.match(r'[0-9]{1,3}.[0-9]{1,3}.[0-9]{1,3}.[0-9]{1,3}', self.host), \
            f'{name} URL host has to be localhost or IP address'
        assert isinstance(self.port, int) and self.port >= 1024 and self.port <= 65_535, \
            f'{name} URL port has to be between 1024 and 65535'
        assert len(self.slug) > 0, f'{name} URL slug has to be specified'
            

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
