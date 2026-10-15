"""Stateful API classes the BFCL multi-turn categories act on.

Vendored from the reference implementation, which is where the benchmark
defines them: an instance's attributes after a turn are what a prediction is
graded against, so a reimplementation would change the scores.

Each class simulates its domain in memory -- there is no file, network or
process access anywhere in them -- and is loaded from a test entry's
``initial_config`` before a rollout begins.

Source: https://github.com/ShishirPatil/gorilla at commit
``ac59a475d3d09dc70ee057008bf178fbc50d9a04``, berkeley-function-call-leaderboard,
``bfcl/eval_checker/multi_turn_eval/func_source_code``. That is the version the
v3 multi-turn ground truth was written against; later versions changed method
signatures the ground truth calls. Only the imports between these modules are
rewritten; the bodies are untouched.
"""

from .gorilla_file_system import GorillaFileSystem
from .math_api import MathAPI
from .message_api import MessageAPI
from .posting_api import TwitterAPI
from .ticket_api import TicketAPI
from .trading_bot import TradingBot
from .travel_booking import TravelAPI
from .vehicle_control import VehicleControlAPI

#: Class name as a test entry's ``involved_classes`` spells it, to the class.
API_CLASSES: dict[str, type] = {
    "GorillaFileSystem": GorillaFileSystem,
    "MathAPI": MathAPI,
    "MessageAPI": MessageAPI,
    "TwitterAPI": TwitterAPI,
    "TicketAPI": TicketAPI,
    "TradingBot": TradingBot,
    "TravelAPI": TravelAPI,
    "VehicleControlAPI": VehicleControlAPI,
}

#: Classes with no state to load, so no scenario is applied to them.
STATELESS_CLASSES: frozenset[str] = frozenset({"MathAPI"})

__all__ = ["API_CLASSES", "STATELESS_CLASSES"]
