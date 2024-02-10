"""
Basic Azure Function App with HTTP Trigger
"""

import logging

import azure.functions as func

from src.geophysical.gmt_toolbox import get_map_section

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(route="gmt_test")
def gmt_test(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP Trigger Function
    """
    logging.info("Python HTTP trigger function processed a request.")
    geo_map = get_map_section(0, 2, 0, 2)
    return func.HttpResponse(
        f"Hello, GMT Test. This HTTP triggered function executed successfully and retrieved a map {geo_map.size}.",
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
