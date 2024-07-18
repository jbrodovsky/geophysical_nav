"""
Flask-based web application for interacting with the geophysical navigation system.
"""

import os
import sqlite3

from flask import Flask, jsonify, render_template, request
from pandas import read_csv
from werkzeug.utils import secure_filename

from src.process_dataset import m77t_to_df

# from geophysical_nav.src.process_dataset import (
#     process_sensor_logger_dataset,
#     save_sensor_logger_dataset,
#     process_mgd77_dataset,
# )


# Add the directory containing process_dataset.py to the path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


app = Flask(__name__)


@app.route("/")
def index():
    tables = get_tables(".db/tracklines.db")
    return render_template("index.html", tables=tables)
    # return jsonify({"message": tables})


# Define a route for uploading a .m77t data file
@app.route("/upload_m77t", methods=["POST"])
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
        # Convert uploaded file to a pandas data frame and format it
        data = read_csv(file, delim_whitespace=True, header=0)
        data = m77t_to_df(data)
        tables = get_tables(".db/tracklines.db")
        if filename not in tables:
            data.to_sql(filename, sqlite3.connect(".db/tracklines.db"), if_exists="replace")
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


# # Define the route for processing sensor logger data
# @app.route("/process_sensor_logger", methods=["POST"])
# def process_sensor_logger():
#     """
#     This route should handle file uploads and call the process_sensor_logger_dataset function
#     For simplicity, we assume files are uploaded with the correct names and a single zip file
#     """
#     if "file" not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400
#     if file:
#         filename = secure_filename(file.filename)
#         file.save(os.path.join("/tmp", filename))
#
#         # Here you would need to unzip the file and process the dataset
#         # For now, we'll just return a success message
#         return jsonify({"message": "File uploaded and processed successfully"}), 200
#
#
# # Define the route for processing MGD77 data
# @app.route("/process_mgd77", methods=["POST"])
# def process_mgd77():
#     """
#     This route should handle file uploads and call the process_mgd77_dataset function
#     For simplicity, we assume files are uploaded with the correct names
#     """
#     if "file" not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400
#     if file:
#         filename = secure_filename(file.filename)
#         file.save(os.path.join("/tmp", filename))
#         # Here you would need to process the dataset
#         # For now, we'll just return a success message
#         return jsonify({"message": "File uploaded and processed successfully"}), 200


# Define a route to download processed files
# @app.route("/download/<path:filename>", methods=["GET"])
# def download_file(filename):
#     """
#     This route should allow users to download processed files
#     """
#     return send_from_directory(directory="/tmp", filename=filename)


if __name__ == "__main__":
    app.run(debug=True)
