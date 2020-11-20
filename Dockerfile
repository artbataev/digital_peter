FROM odsai/python-gpu

USER root
RUN apt-get update && apt-get install -y git build-essential g++ make python3.7-dev libopenmpi-dev
RUN pip install albumentations editdistance tqdm
RUN pip install git+https://github.com/parlance/ctcdecode.git

# original user from odsai/python-gpu
USER webapp
