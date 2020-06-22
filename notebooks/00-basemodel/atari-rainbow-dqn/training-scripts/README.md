## Automatic Startup on Google Cloud
Open up the Google Cloud Shell, copy/paste the following and press ENTER:

```shell script
gcloud compute instances create fom-training \
  --zone=europe-west2-a \
  --machine-type=n1-standard-8 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --metadata="install-nvidia-driver=True",startup-script='#! /bin/bash
gsutil cp gs://fom-uni-project/00-autotrain/startup.sh startup.sh
sudo useradd -m -s /bin/bash training
cp startup.sh /home/training/
chmod +x /home/training/startup.sh
sudo -u training bash -c "cd ~/; ./startup.sh pong nors >/dev/null 2>&1 &"
EOF'
```

In this case, you will spin-up a GCloud Instance which will start the training 
of pong without rewards shaping based on the Script pong_nors.sh in the directory
training-scripts.

You can create other scripts and use this to start them up. Just remember the following:
In the second last line you need to change the name of the game and the mode after "./startup.sh".
The scripts you create need to be named like <game>_<mode>.sh (i.e. pong_rs1.sh).