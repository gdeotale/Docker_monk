FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3.8
RUN : \
    && apt-get update \
    && apt-get install -y git \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends python3.8-venv \
    && apt-get install libpython3.8-dev -y \
    && apt-get clean \
    && :
	
# Add env to PATH
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install ASAP
RUN : \
    && apt-get update \
    && apt-get -y install curl \
    && curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1-(Nightly)/ASAP-2.1-Ubuntu2004.deb" \
    && dpkg --install ASAP-2.1-Ubuntu2004.deb || true \
    && apt-get -f install --fix-missing --fix-broken --assume-yes \
    && ldconfig -v \
    && apt-get clean \
    && echo "/opt/ASAP/bin" > /venv/lib/python3.8/site-packages/asap.pth \
    && rm ASAP-2.1-Ubuntu2004.deb \
    && :
	
# Install OpenSlide dependencies
RUN : \
    && apt-get update \
    && apt-get install -y openslide-tools libopenslide0 \
    && apt-get install -y build-essential libffi-dev libxml2-dev libjpeg-turbo8-dev zlib1g-dev \
    && apt-get clean \
    && :

# Install OpenSlide Python bindings
RUN /venv/bin/python3.8 -m pip install --no-cache-dir openslide-python

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# update permissions
RUN chown -R user:user /venv/

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

# Update pip
RUN /venv/bin/python3.8 -m pip install pip --upgrade


# You can add any Python dependencies to requirements.txt
RUN /venv/bin/python3.8 -m pip install \
    --no-cache-dir \
    -r /opt/app/requirements.txt

#install pytorch
RUN /venv/bin/python3.8 -m pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
#RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN /venv/bin/python3.8 -m pip install ultralytics>=8.2.4
# Verify torch installation to ensure it's available
RUN /venv/bin/python3.8 -c "import torch; print(torch.__version__)"

# Install MMEngine and MMCV
RUN pip install openmim && \
    mim install "mmengine==0.7.1" "mmcv==2.0.0rc4"
	
RUN mim install mmdet

COPY ./mmdetection /mmdetection

RUN pip install yapf==0.40.1
RUN pip install setuptools==59.5.0

COPY ./config/dino_swin_monkey1.py /mmdetection/configs/dino/
COPY ./config/dino_swin_monkey2.py /mmdetection/configs/dino/

COPY --chown=user:user structures.py /opt/app/
COPY --chown=user:user ./utils /opt/app/utils 
COPY --chown=user:user ./wsdetectron2.py /opt/app/ 
COPY --chown=user:user ./models /opt/app/models

COPY --chown=user:user model/Dino1.pth /opt/ml/model1/
COPY --chown=user:user model/Dino2.pth /opt/ml/model1/
COPY --chown=user:user model/yolov5.pth /opt/ml/model1/

COPY --chown=user:user ./export.py /opt/app/ 

RUN pip install wholeslidedata
RUN /venv/bin/python3.8 -m pip install 'git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@main'

RUN pip install numba

COPY --chown=user:user ensemble_boxes/ensemble_boxes_nms.py /opt/app/ 
COPY --chown=user:user utils_ensemble.py /opt/app/
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user visualizer.py /venv/lib/python3.8/site-packages/mmengine/visualization/visualizer.py
COPY --chown=user:user registry.py /venv/lib/python3.8/site-packages/mmengine/registry/registry.py 

USER user
ENTRYPOINT ["/venv/bin/python3.8", "inference.py"]
