from data_management import dbmgr

dbmgr.say_hello()

from navtoolbox import say_hello

say_hello()

from navtoolbox.earth import say_hello as hello_earth

hello_earth()

from navtoolbox import transform

transform.say_hello()

from pyins import _numba_integrate as nbi
import numpy as np

rv = np.array([90, 0, 0])
M = np.empty((3, 3))

nbi.mat_from_rotvec(rv, M)
print(f"{M=}")

print(f"{transform.mat_from_rotvec(rv)=}")
