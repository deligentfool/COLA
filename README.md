# COLA

## Note
COLA is the first work to address the Dec-POMDP problem by integrating contrastive learning into multi-agent reinforcement learning. And we believe the COLA framework is the most cost-effective, bringing remarkable performance improvement with minor changes of reinforcement learning models. Our approach of simply adding a one-hot consensus encoding to the network input can be extended to any other multi-agent reinforcement learning algorithm.

The implementation of the following methods can also be found in this codebase, which are finished by the authors of [PyMARL](https://github.com/oxwhirl/pymarl). PyMARL is written in PyTorch and uses [SMAC](https://github.com/oxwhirl/smac) as its environment.

## Installation instructions

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

It is worth noting that we run the all experiments on **SC2.4.6.2.69232**, not easier SC2.4.10. Performance is *not* always comparable between versions.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

Set up Google Research Football: Please follow the Quick Start in https://github.com/google-research/football.

## Run an experiment 

```shell
python3 src/main.py --config=cola --env-config=sc2 with env_args.map_name=2s3z env_args.seed=1
```

or
```shell
python3 src/main.py --config=cola --env-config=academy_3_vs_1_with_keeper with seed=1
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.

The previous config files used for the SMAC Beta have the suffix `_beta`.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.
