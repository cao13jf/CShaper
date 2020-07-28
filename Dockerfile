FROM ufoym/deepo

MAINTAINER jeff jfcao3-c@my.cityu.edu.hk

RUN mkdir -p /home/CShaper/CShaper \
    && cd /home/CShaper/CShaper \
    && git clone --depth 1 https://github.com/cao13jf/CShaper.git \

WORKDIR ./
