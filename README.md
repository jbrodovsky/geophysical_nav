# Geophysical Navigation

Toolbox for INS aiding via geophysical position feedback. Restructuring my old research repo to work here instead.

## Working notes

### Update 14 May 2024

So I'm going back to school full time or having this be a part of my greater ARL research, but I'm going back to simpler yet somewhat less elegant approach. Instead of focusing on a cloud based implementation I'm going to implement a local database and "server-lite" approach where I can develop the code on any machine, then pull down the most recent version from `main` and run it on a separate computer in potentially a headless fashion.

### Old notes

* Planning and thoughts
  * Restructured the project to follow more standard Python and PyPI practices.
    * Using `setuptools` and `pybind11` I can write back-end C++ code if needed and build and install the module using `pip install .`.
    * `pyins` is still somewhat of a issue since it is not available on PyPI, but I have scripts written to deal with it a the momement, Azure deployment still needs to be dealt with in this manner.
    * Given Azure's preference for `pip` and `pygmt` now being available via pip, there isn't the functional need for `conda` right now.
      * Addendum: `pygmt` still only works properly with a `conda` install. Can manage a conda environment for Azure Functions by developing the app in a container.
      * Azure functions works perfectly fine when manually run from the command line within a `conda` environment: `func host start`.
      * Manual launch and conda environments seems to block the debugger
  * Thought process right now is to deliver two things
    * A set of CLI tools via `project.scripts` in the `pyproject.toml` file that can be built using PIP. Intent for these is local debugging and small scale testing
    * A serverless web app that provides cloud functionality and hooks for the same functionality but in the cloud. Intent for that is for large-scale processing and testing
* Current workflow:
  1. Download raw `.m77t` files NOAA
  2. Using notebook, run the *Preprocess tracklines into sql/df format* section clean up the data formate
  3. Using notebook, run the *Parse the raw data into tracklines of continuous data collections
  4. Run particle filter sim (`bathy_pf.py`) to processes the data and run post-processing results
* Tentative ideal future workflow has steps 1-4 above as serverless functions. Steps 2-3 would run automatically on new uploads of `.m77t` data. Step 4 would run on command.
  