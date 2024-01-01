"""
Tools for interacting with the database.
"""

import sqlite3
import os

from flask import request, jsonify, Blueprint, render_template
from pandas import read_csv
from werkzeug.utils import secure_filename

from ..src.process_dataset import m77t_to_df

up = Blueprint("db", __name__, url_prefix="/db")


def get_tables(db_path: str):
    """
    Get the names of all tables in a database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    conn.close()

    # The result is a list of tuples. Convert it to a list of strings.
    tables = [table[0] for table in tables]

    return tables


# Define a route for uploading a .m77t data file
@up.route("/upload", methods=["GET"])
def upload():
    return render_template("/db/upload.html")


@up.route("/upload_m77t", methods=["POST"])
def upload_m77t():
    """
    This route should handle file uploads and call the save_sensor_logger_dataset function
    For simplicity, we assume files are uploaded with the correct names
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        tables = get_tables(".db/tracklines.db")
        if filename not in tables:
            # Convert uploaded file to a pandas data frame and format it
            data = read_csv(file, delimiter="\t", header=0)
            data = m77t_to_df(data)
            data.to_sql(
                filename, sqlite3.connect(".db/tracklines.db"), if_exists="replace"
            )
        else:
            return jsonify({"error": "File already exists"}), 400

        file.save(os.path.join("/tmp", filename))
        # Here you would need to process the dataset
        # For now, we'll just return a success message
        return jsonify({"message": "File uploaded and processed successfully"}), 200

    return (
        jsonify({"error": "Something went wrong, intended pathes not executed."}),
        400,
    )


# def get_db():
#     if "db" not in g:
#         g.db = sqlite3.connect(
#             current_app.config["DATABASE"], detect_types=sqlite3.PARSE_DECLTYPES
#         )
#         g.db.row_factory = sqlite3.Row
#
#     return g.db
#
#
# def close_db(e=None):
#     db = g.pop("db", None)
#
#     if db is not None:
#         db.close()
#
#
# def init_db():
#     db = get_db()
#
#     with current_app.open_resource("schema.sql") as f:
#         print("schema opened")
#         db.executescript(f.read().decode("utf8"))
#
#
# @click.command("init-db")
# def init_db_command():
#     """Clear the existing data and create new tables."""
#     init_db()
#     click.echo("Initialized the database.")
#
#
# def init_app(app):
#     app.teardown_appcontext(close_db)
#     app.cli.add_command(init_db_command)
#
