# To enable ssh & remote debugging on app service change the base image to the one below
FROM mcr.microsoft.com/azure-functions/python:4-python3.11-appservice

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true

COPY requirements.txt /requirements.txt
COPY environment.yml /environment.yml
# RUN pip install -r /requirements.txt

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-1-Linux-x86_64.sh -O conda-install.sh && \
    chmod 755 conda-install.sh && \
    sh conda-install.sh -b -p $HOME/miniconda && \
    rm -f conda-install.sh

ENV PATH="${HOME}/miniconda/bin:${PATH}"

RUN conda init bash && \
    conda env update --name base --file /environment.yml && \
    pip install -r /requirements.txt

COPY . /home/site/wwwroot