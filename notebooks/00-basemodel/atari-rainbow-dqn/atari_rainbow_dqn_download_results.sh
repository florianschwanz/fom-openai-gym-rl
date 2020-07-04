USER=training
HOST=34.105.211.174
RUN_TO_LOAD=2020-06-21-14:26:22

rm -rf ./output/${RUN_TO_LOAD}
mkdir -p ./output/${RUN_TO_LOAD}

scp ${USER}@${HOST}:~/fom-openai-gym-rl/notebooks/00-basemodel/atari-rainbow-dqn/output/${RUN_TO_LOAD}/* ./output/${RUN_TO_LOAD}/