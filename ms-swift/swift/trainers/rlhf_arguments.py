from dataclasses import dataclass, field
from typing import List, Optional

from trl import CPOConfig as HfCPOConfig
from trl import DPOConfig as HfDPOConfig
from trl import GRPOConfig as HfGRPOConfig
from trl import KTOConfig as HfKTOConfig
from trl import ORPOConfig as HfORPOConfig
from trl import PPOConfig as HfPPOConfig
from trl import RewardConfig as HfRewardConfig

from .arguments import GRPOArgumentsMixin, SwiftArgumentsMixin


@dataclass
class DPOConfig(SwiftArgumentsMixin, HfDPOConfig):
    pass


@dataclass
class CPOConfig(SwiftArgumentsMixin, HfCPOConfig):
    pass


@dataclass
class ORPOConfig(SwiftArgumentsMixin, HfORPOConfig):
    pass


@dataclass
class KTOConfig(SwiftArgumentsMixin, HfKTOConfig):
    pass


@dataclass
class RewardConfig(SwiftArgumentsMixin, HfRewardConfig):
    pass


@dataclass
class PPOConfig(SwiftArgumentsMixin, HfPPOConfig):
    pass


@dataclass
class GRPOConfig(GRPOArgumentsMixin, SwiftArgumentsMixin, HfGRPOConfig):
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.
    stop_words: List[str] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if self.cosine_max_len is None:
            self.cosine_max_len = self.max_completion_length
