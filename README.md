# facenet-playground

Repo with code for FaceNet 
([paper](https://arxiv.org/abs/1503.03832)) and RetinaFace 
([paper](https://arxiv.org/abs/1905.00641)) usage.

## Repo structure

- `data` - contains trained models (stored with git LFS)
- `scripts` - contains all useful scripts:
    - `common.py` - common used functions
    - `cut_faces.py` - optional script for faces preparation (cutting)
    - `demo.py` - demonstration script with main functional
    - `evaluate.py` - script for .tsv files evaluation
    - `face_detector.py` - helper class to use 
    [RetinaFace](https://github.com/peteryuX/retinaface-tf2) model
    - `face_net.py` - helper class to use 
    [FaceNet](https://github.com/davidsandberg/facenet) model

## Installation

All required python dependencies could be found in `requirements.txt`

### Step-by-step installation in docker + venv
1. Can recommend to use my [docker image](https://github.com/resolator/rdi) for 
further installation:
```bash
docker run \
--gpus all \
--shm-size 8G \
-e TERM=$TERM \
-e UNAME=$(whoami) \
-e UID=$(id -u) \
-e GID=$(id -g) \
-e DISPLAY=unix$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /etc/localtime:/etc/localtime:ro \
-v $HOME/:/host_home \
--name facenet-playground \
-it \
resolator/rdi:cuda-10.1-cudnn7-runtime-ubuntu18.04
```

2. Clone this repo into docker container:
```bash
git clone https://github.com/resolator/facenet-playground.git
```

3. Run `install.sh` and activate prepared venv:
```bash
cd facenet-playground
./install.sh
source .venv/bin/activate
```

4. Enjoy!

## Usage

Use `<script_name>.py --help` to see scripts usages.

For evaluation by yourself use `scripts/evaluate.py`. For example:
```bash
./scripts/evaluate.py --gt-tsv data/gt.tsv --pd-tsv data/pd.tsv
```


## Evaluation results

- 2020-08-09: Accuracy increased to 76.4%. Achieved with the following command:
```bash
./scripts/demo.py --database-dir /data/facenet/datasets/mark/ --testing-dir /data/facenet/datasets/test/ --save-to /data/facenet/datasets/tmp --bs
250 --space l2 --img-size 160 160 --expand-factors 0.25 0.2 0.25 0.05
```

- 2020-07-28: Current accuracy: 67.88%.
