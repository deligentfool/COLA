from functools import partial
from .starcraft2.starcraft2 import StarCraft2Env
from .multiagentenv import MultiAgentEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

#if sys.platform == "linux":
#    os.environ.setdefault("SC2PATH",
#                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join("/", "home", "xzw", "3rdparty", "StarCraftII"))