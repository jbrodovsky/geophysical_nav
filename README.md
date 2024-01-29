# Geophysical Navigation

Toolbox for INS aiding via geophysical position feedback. Restructuring my old research repo to work here instead.

## Working notes

* Restructured virtual environment maintainer: `pygmt` is now available via `pip`
  * While I kind of prefer `conda`'s method for virtual environment maintenance, between using `pyproject.toml` files to build C++, GMT availble via `pip`, Azure prefering `pip`, I can't really justify using `conda` right now.
* Restructured the workflow. Flask app was a nice portfolio project, but in reality all I needed was to use an ssh tunnel to the remote machine
  * Update: old remote machine is out of commission for time being. Can run reasonably quickly on laptop with multiprocessing, best bet is to covert over to a serverless application (leaning Azure)
* Current workflow:
  1. Download raw `.m77t` files NOAA
  2. Using notebook, run the *Preprocess tracklines into sql/df format* section clean up the data formate
  3. Using notebook, run the *Parse the raw data into tracklines of continuous data collections
  4. Run particle filter sim (`bathy_pf.py`) to processes the data and run post-processing results
* Tentative ideal future workflow has steps 1-4 above as serverless functions. Steps 2-3 would run automatically on new uploads of `.m77t` data. Step 4 would run on command.
  