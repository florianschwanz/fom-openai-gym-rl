#! /bin/bash
export RUN_NAME="freeway_rs_both_emph_hori"

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
export NUM_FRAMES=100_000

RANDOM_NUMBERS=$(awk -v n=10 -v seed="$RANDOM" 'BEGIN { srand(seed); for (i=0; i<n; ++i) printf("%.1f\n", rand()) }')
RAND1=$(echo $RANDOM_NUMBERS | awk '{ print $1 }')
RAND2=$(echo $RANDOM_NUMBERS | awk '{ print $2 }')

export REWARD_FREEWAY_DISTANCE_WALKED=${RAND1}
export REWARD_FREEWAY_DISTANCE_TO_CAR=${RAND2}
export REWARD_POTENTIAL_BASED=0.0

./gcloud_atari_rainbow_dqn.py &
