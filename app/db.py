"""
Tools for interacting with the database. Raw data processing and parsing tools
should be in src/process_dataset.py
"""

import sqlite3

from flask import request, jsonify, Blueprint, render_template
from pandas import read_csv, read_sql_query
from werkzeug.utils import secure_filename

from ..src.process_dataset import (
    m77t_to_df,
    parse_tracklines_from_db,
    save_dataset,
    get_tables,
    get_parsed_data_summary,
)

up = Blueprint("db", __name__, url_prefix="/db")


# Define a route for uploading a .m77t data file
@up.route("/upload", methods=["GET"])
def upload():
    """Route for raw .csv file upload"""
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
        filename = "m77t_" + filename.split(".")[0]
        tables = get_tables(".db/tracklines.db")
        if filename not in tables:
            # Convert uploaded file to a pandas data frame and format it
            data = read_csv(file, delimiter="\t", header=0)
            data = m77t_to_df(data)
            # data.to_sql(
            #    filename, sqlite3.connect(".db/tracklines.db"), if_exists="replace"
            # )
            save_dataset(
                [data],
                [filename],
                output_location=".db",
                output_format="db",
                dataset_name="tracklines",
            )
        else:
            return jsonify({"error": "File already exists"}), 400

        # file.save(os.path.join("/tmp", filename))
        # Here you would need to process the dataset
        # For now, we'll just return a success message
        return jsonify({"message": "File uploaded and processed successfully"}), 200

    return (
        jsonify({"error": "Something went wrong, intended pathes not executed."}),
        400,
    )


# define a route for the blueprint to use parse_tracklines_from_db
@up.route("/parse", methods=["GET"])
def parse():
    """
    Check if the database .db/parsed.db exists, if it does, get the summary table
    """
    # Check if the database exists
    tables = get_tables(".db/parsed.db")
    print(request)
    if "summary" in tables:
        # read in the summary table using pandas
        with sqlite3.connect(".db/parsed.db") as conn:
            summary = read_sql_query("SELECT * FROM summary", conn)
    else:
        summary = None
    return render_template("/db/parse.html", summary=summary)


@up.route("/parse_tracklines", methods=["POST"])
def parse_tracklines():
    """
    Route for parsing the tracklines from the database
    """
    max_time = float(request.form.get("max_time"))
    max_delta_t = float(request.form.get("max_delta_t"))
    min_duration = float(request.form.get("min_duration"))

    try:
        tables = get_tables(".db/tracklines.db")
    except FileNotFoundError:
        return (
            jsonify({"error": "No tables in database or database not initialized"}),
            400,
        )

    # Convert uploaded file to a pandas data frame and format it
    data, names = parse_tracklines_from_db(
        ".db/tracklines.db",
        max_time,
        max_delta_t,
        min_duration,
        data_types=[
            "bathy",
            "mag",
            "grav",
            ["bathy", "mag"],
            ["bathy", "grav"],
            ["grav", "mag"],
            ["bathy", "grav", "mag"],
        ],
    )
    # Save the parsed data to the database
    save_dataset(
        data, names, output_location=".db", output_format="db", dataset_name="parsed"
    )
    summary = get_parsed_data_summary(data, names)
    save_dataset(
        summary,
        ["summary"],
        output_location=".db",
        output_format="db",
        dataset_name="parsed",
    )
    return (
        jsonify(
            {
                "message": "Tracklines parsed successfully with the following specs: "
                + f"max time={max_time}, max delta t={max_delta_t}, min duration={min_duration}",
                "new_tables": names,
            }
        ),
        200,
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
