#!/bin/bash
python --version
pip install azure-cli --force-reinstall
pip install --upgrade azureml-sdk[cli]
pip install azureml-inference-server-http