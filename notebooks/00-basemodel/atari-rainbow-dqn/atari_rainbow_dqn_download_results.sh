USER=training
HOST=35.189.113.44
RUN_TO_LOAD=2020-07-03-11:21:07

# rm -rf ./output/${RUN_TO_LOAD}
mkdir -p ./output/${RUN_TO_LOAD}

#scp ${USER}@${HOST}:~/fom-openai-gym-rl/notebooks/00-basemodel/atari-rainbow-dqn/output/${RUN_TO_LOAD}/*.pickle ./output/${RUN_TO_LOAD}/
scp ${USER}@${HOST}:~/fom-openai-gym-rl/notebooks/00-basemodel/atari-rainbow-dqn/output/${RUN_TO_LOAD}/*.gif ./output/${RUN_TO_LOAD}/
scp ${USER}@${HOST}:~/fom-openai-gym-rl/notebooks/00-basemodel/atari-rainbow-dqn/output/${RUN_TO_LOAD}/*rewards.png ./output/${RUN_TO_LOAD}/
scp ${USER}@${HOST}:~/fom-openai-gym-rl/notebooks/00-basemodel/atari-rainbow-dqn/output/${RUN_TO_LOAD}/*losses.png ./output/${RUN_TO_LOAD}/
scp ${USER}@${HOST}:~/fom-openai-gym-rl/notebooks/00-basemodel/atari-rainbow-dqn/output/${RUN_TO_LOAD}/*.txt ./output/${RUN_TO_LOAD}/
scp ${USER}@${HOST}:~/fom-openai-gym-rl/notebooks/00-basemodel/atari-rainbow-dqn/output/${RUN_TO_LOAD}/*.csv ./output/${RUN_TO_LOAD}/