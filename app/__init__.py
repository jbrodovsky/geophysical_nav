"""
GUI for geophysical navigation
"""

import os

from flask import Flask, render_template, jsonify
from . import home, db


def create_app(test_config=None):
    """create and configure the app"""
    # Boiler plate code from Flask tutorial
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY="dev",
        DATABASE=os.path.join(app.instance_path, "flaskr.sqlite"),
    )
    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)
    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    # End boiler plate code from Flask tutorial

    app.register_blueprint(home.bp)
    app.add_url_rule("/", endpoint="index")

    app.register_blueprint(db.up)
    # app.add_url_rule("/upload", endpoint="upload")

    # Place holder routes
    # @app.route("/run")
    # def run():
    #    return jsonify({"message": "Run route"})

    # @app.route("/results")
    # def results():
    #    return jsonify({"message": "results route"})

    return app
