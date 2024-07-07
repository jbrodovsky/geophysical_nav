"""
Basic Azure Function App with HTTP Trigger
"""

import os
import uuid
import logging

import azure.functions as func

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


from src.geophysical.gmt_toolbox import get_map_section, get_map_point

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(route="gmt_test")
def gmt_test(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP Trigger Function
    """
    logging.info("Python HTTP trigger function processed a request.")
    geo_map = get_map_section(0, 2, 0, 2)
    value = get_map_point(geo_map, 1, 1)
    return func.HttpResponse(
        f"GMT Test: This function executed successfully and retrieved a map value: {value}.",
        status_code=200,
    )


@app.route(route="http_trigger")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP Trigger Function
    """
    logging.info("Python HTTP trigger function processed a request.")

    name = req.params.get("name")
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get("name")

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully."
            + " Pass a name in the query string or in the request body for a personalized response.",
            status_code=200,
        )


@app.route(route="HttpQueueExample")
@app.queue_output(arg_name="msg", queue_name="outqueue", connection="AzureWebJobsStorage")
def queue_output(req: func.HttpRequest, msg: func.Out[func.QueueMessage]) -> func.HttpResponse:
    """
    HTTP Trigger Function with Queue Output
    """
    logging.info("Python HTTP trigger function processed a request.")

    name = req.params.get("name")
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get("name")

    if name:
        msg.set(name)
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. "
            + "Pass a name in the query string or in the request body for a personalized response.",
            status_code=200,
        )


@app.route(route="create_container")
def create_container(req: func.HttpRequest) -> func.HttpResponse:
    """
    Create a blob container with req.name
    """
    logging.info("Python HTTP trigger function processed a request.")

    name = req.params.get("name")
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get("name")

    if name:
        # Create a unique name for the container
        container_name = str(uuid.uuid4())

        # Create the BlobServiceClient object which will be used to create a container client
        blob_service_client = BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])

        # Create a new container
        container_client = blob_service_client.create_container(container_name)

        return func.HttpResponse(f"Container {container_name} created successfully.")
    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. "
            + "Pass a name in the query string or in the request body for a personalized response.",
            status_code=200,
        )


# try:
#    print("Azure Blob Storage Python quickstart sample")
#
#    # Quickstart code goes here
#
# except Exception as ex:
#    print("Exception:")
#    print(ex)
#
