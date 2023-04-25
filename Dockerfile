# FROM dp-hpc-registry.cn-wulanchabu.cr.aliyuncs.com/eflops/pytorch2.0:py3.10
FROM python:3.8
# RUN apt update && apt install -y --no-install-recommends openssh-server sudo
# RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/g' /etc/ssh/sshd_config
# RUN pip install ipykernel
COPY . .
RUN pip install -e .
