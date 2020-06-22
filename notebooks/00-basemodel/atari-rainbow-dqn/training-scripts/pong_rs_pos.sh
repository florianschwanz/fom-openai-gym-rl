export RUN_NAME="pong_rs_pos"

export OUTPUT_DIRECTORY="$HOME/fom-openai-gym-rl/notebooks/00-basemodel/atari-rainbow-dqn/output/"
export ENVIRONMENT_ID="PongNoFrameskip-v4"
export BATCH_SIZE=32
export GAMMA=0.99

export EPS_START=1.0
export EPS_END=0.01
export EPS_DECAY=10_000

export NUM_ATOMS=51
export VMIN=-10
export VMAX=10

export TARGET_UPDATE=1_000
export REPLAY_MEMORY_SIZE=100_000
export NUM_FRAMES=3_000_000

export REWARD_PONG_PLAYER_RACKET_COVERS_BALL=0.5

cd ..
./gcloud_atari_rainbow_dqn.py &
