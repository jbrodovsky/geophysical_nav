# Geophysical Navigation
Toolbox for INS aiding via geophysical position feedback. Restructuring my old research repo to work here instead.

**Working notes**

* Restructured the workflow. Flask app was a nice portfolio project, but in reality all I needed was to use an ssh tunnel to the remote machine
* New workflow:
  * Develop locally on laptop and run tests
  * Configure simulations to run in either a notebook or from the command line
  * Manually or via git transfer the data files (sql or m77t is up for negotiation, sql binaries are actually smaller, probably don't want to just blindly trust that the sql files aren't tampered with on the internet)
    * -> Manually transfer results using `scp`
  * Test trajectory processing locally, run actual experiments on server
  