export RUN_NAME="pong_nors"

export TELEGRAM_CONFIG_FILE="telegram.config"
export ENVIRONMENT_ID="FreewayNoFrameskip-v4"
export BATCH_SIZE=32
export LEARNING_RATE=0.0001
export GAMMA=0.99

export EPS_START=1.0
export EPS_END=0.01
export EPS_DECAY=10_000

export NUM_ATOMS=51
export VMIN=-10
export VMAX=10

export TARGET_UPDATE=1_000
export REPLAY_MEMORY_SIZE=100_000
export NUM_FRAMES=1_000_000

export REWARD_FREEWAY_DISTANCE_WALKED=0.0
export REWARD_FREEWAY_DISTANCE_TO_CAR=0.5
export REWARD_POTENTIAL_BASED=0.0

./gcloud_atari_rainbow_dqn.py &
