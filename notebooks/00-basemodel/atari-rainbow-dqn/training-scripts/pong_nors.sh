#! /bin/bash
export RUN_NAME="pong_nors"

export TELEGRAM_CONFIG_FILE="telegram.config"
export ENVIRONMENT_ID="PongNoFrameskip-v4"
export BATCH_SIZE=32
export LEARNING_RATE=0.0001
export GAMMA=0.99

export EPS_START=1.0
export EPS_END=0.01
export EPS_DECAY=10_000

export NUM_ATOMS=51
export VMIN=-10
export VMAX=10

export NORMALIZE_SHAPED_REWARD=False
export REWARD_SHAPING_DROPOUT_RATE=0.0

export TARGET_UPDATE_RATE=10
export MODEL_SAVE_RATE=10
export EPISODE_LOG_RATE=10

export REPLAY_MEMORY_SIZE=100_000
export NUM_FRAMES=3_000_000

./gcloud_atari_rainbow_dqn.py &
