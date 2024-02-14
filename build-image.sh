#/bin/bash
az login
az acr build --registry geonavcontainers --image geonav:v0.1.0 .