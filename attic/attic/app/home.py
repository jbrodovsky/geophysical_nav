"""
Home page definition
"""

from flask import Blueprint, render_template

from .db import get_tables

bp = Blueprint("home", __name__)


@bp.route("/")
def index():
    """Define home page."""
    tables = get_tables(".db/tracklines.db")
    return render_template("home/index.html", tables=tables)
