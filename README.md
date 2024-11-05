# Geophysical Navigation

Toolbox for INS aiding via geophysical position feedback. Restructuring my old research repo to work here instead.

## Working notes

### Update 5 Noveber 2024

**Dev environment**

Got fed up with `conda` and switched to `pixi` for environment and package management. Pixi builds on the conda philosophy but instead of having a central local environment it instead builds the environment within the project folder. This requires a slightly different workflow:

1. Project dependencies are listed in the `pyproject.toml` file under the `[tool.pixi]` section. I've additionally seperated out the dependancies for testing, linting, and development into seperate sections. This requires that the appropriate environment be 'activated' (namely `dev` since I use notebooks and other interactive tools in testing) using `pixi shell -e dev`.
2. The project does not need to build manually built prior to writing an experiment in `/scripts`. Pixi lists the current project as an editable dependancy and builds and installs the project in the environment when the shell and environment is activated using `pixi shell`. Note that the root folder/project name `geophysical_nav` is not required anymore and imports and intellisense works from below the `/src` folder.

**Language usage**

I'm getting a little annoyed wrestling with `numba` and don't like storing the navigation states in arrays which require a known order of states. The arrays are somewhat useful for the linear algebra computations they enable, but that can likely be stored and utilized still with small specific sets of states ex: `nav_states.position = [lat, lon, alt]` and `nav_states.velocity = [v_n, v_e, v_d]`. I'm going to see about using Numba's `jitclass` functionality. If that gets annoying, I'll switch to using `pybind11` to write the backend in C++ and use Python as a scripting language to run the simulations and manage the data.

The main issue is that I don't want to write *multiple* versions of the basic strapdown integration method. For the Kalman filter approaches, it's all the same and can easily be done with matrix math. For the particle filter implementations, I can use that existing integration method and *loop* over all the particles (tried that in Python, its ***slow***) or I can vectorize the calculations and write another integration method that will need to be tested against the prior one. Converting the existing method to a C/C++ backend will allow me to meet somewhere in the middle. I can write a single integration method that only needs to be tested once, and can take advantage of compilation to speed up the looped calculations.

### Update 30 August 2024

Using Insync and OneDrive I've gotten around the source data issue. The plan is to store the source data and resulting processed database on OneDrive location and use this a network drive. Insync allows for OneDrive syncing directly to the filesystem on a linux machine and will allow me to transfer and access the data on my remote linux desktop. Development paradigm will be to use my laptop and WSL2 for development and testing. Code will then be pushed to GitHub and pulled down on the remote desktop for testing and deployment and running full-scale simulations.

To that end, I need to look into packaging the source code and library a bit more. I think the current model of packaging it as a conda environment is the best place to start from. 

### Update 16 July 2024

So I feel that this project is a good enough excuse for me to work on my C/C++ skills as well. Also considering the sheer bulk of data and repetitions I'm going to be making, having a faster backend would be beneficial. I'm going to be using `pybind11` to write the backend code in C++ and then use Python to interface with it. All Python-based backend code should make use of heavy type hints and be transcompiled and built using MyPyC. Simulations and experiments should be conducted in the `scripts` folder and make use of the built source code found in `src`.

The general plan is to use Python more like a highly developed scripting and shell language to run the simulations and manage the data, while the heavy lifting is done in C++. Data pre- and post-processing will be done in Python while the key navigation algorithm code and run-to-run simulation will be done in C++. I'm not yet sure how to best make use of multi threading or parallel processing. The main issue is passing the trajectory information between the two languages. I can convert the DataFrame to a NumPy array and pass that over to C++, but then it's Python running the simulation and I can't make use of C++ proper multi-threading. Alternatively I can rebuild a sqlite client in C++ and have it read the selected data from the database but that seems like a lot of work for little gain. I'll have to think about this more.



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
  