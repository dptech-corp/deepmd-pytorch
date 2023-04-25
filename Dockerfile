FROM dp-hpc-registry.cn-wulanchabu.cr.aliyuncs.com/eflops/pytorch2.0:py3.10
COPY . .
RUN pip install . -i https://pypi.org/simple