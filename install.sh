sudo apt update && sudo apt install -y git-lfs python3-dev python3-venv python3-opencv
git-lfs pull
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

