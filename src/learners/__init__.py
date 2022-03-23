from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .cq_learner import CQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["cq_learner"] = CQLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
