from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .cola_learner import COLALearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["cola_learner"] = COLALearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
