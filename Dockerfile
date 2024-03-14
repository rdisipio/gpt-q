FROM registry.sdcc.bnl.gov/sciserver/gpu-essentials:2.0a

USER idies

WORKDIR /home/idies

COPY environment.yml /home/idies
COPY requirements-no-version.txt /home/idies

RUN ( conda env create --name qml --file=environments.yml )

USER root
