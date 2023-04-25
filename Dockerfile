FROM dp-hpc-registry.cn-wulanchabu.cr.aliyuncs.com/eflops/pytorch2.0:py3.10
RUN apt update && \
    apt install -y --no-install-recommends \
    openssh-server \
    sudo
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/g' /e
tc/ssh/sshd_config
RUN pip install ipykernel
COPY . .
RUN pip install tqdm
RUN pip install h5py
RUN pip install wandb
RUN pip install . -i https://pypi.org/simple