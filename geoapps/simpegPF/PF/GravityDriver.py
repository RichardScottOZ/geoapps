import re
import os
from .. import Mesh, Utils
import numpy as np
from . import BaseGrav, Gravity


class GravityDriver_Inv:
    """docstring for GravityDriver_Inv"""

    def __init__(self, input_file=None):
        if input_file is not None:
            self.basePath = os.path.sep.join(input_file.split(os.path.sep)[:-1])
            if len(self.basePath) > 0:
                self.basePath += os.path.sep
            self.readDriverFile(input_file.split(os.path.sep)[-1])

    def readDriverFile(self, input_file):
        """
            Read input files for forward modeling GRAV data with integral form
            INPUT:
            input_file: File name containing the forward parameter

            OUTPUT:
            mshfile
            obsfile
            topofile
            start model
            ref model
            active cells model
            weightfile
            chi_target
            as, ax ,ay, az
            upper, lower bounds
            lp, lqx, lqy, lqz
            eps_p, eps_q
            # All files should be in the working directory,
            # otherwise the path must be specified.

        """

        fid = open(self.basePath + input_file)

        # Line 1: Mesh
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        if l_input[0] == "FILE":
            mshfile = l_input[1].rstrip()
        else:
            mshfile = None

        # Line 2: Observation file
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        obsfile = l_input[1].rstrip()

        # Line 3: Topo, active-dyn, active-static
        topofile = None
        staticInput = None

        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        if l_input[0] == "TOPO":
            topofile = l_input[1].rstrip()

        elif l_input[0] == "VALUE":
            staticInput = float(l_input[1])

        elif l_input[0] == "FILE":
            staticInput = l_input[1].rstrip()

        # Line 4: Starting model
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        if l_input[0] == "VALUE":
            mstart = float(l_input[1])

        elif l_input[0] == "FILE":
            mstart = l_input[1].rstrip()

        # Line 5: Reference model
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        if l_input[0] == "VALUE":
            mref = float(l_input[1])

        elif l_input[0] == "FILE":
            mref = l_input[1].rstrip()

        # Line 6: Cell weights
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        if l_input[0] == "DEFAULT":
            wgtfile = None

        elif l_input[0] == "FILE":
            wgtfile = l_input[1].rstrip()

        # Line 7: Target chi-factor
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        if l_input[0] == "DEFAULT":
            chi = 1.0

        elif l_input[0] == "VALUE":
            chi = float(l_input[1])

        # Line 8: Alpha values
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        if l_input[0] == "VALUE":

            val = np.array(l_input[1:5])
            alphas = val.astype(np.float)

        elif l_input[0] == "DEFAULT":

            alphas = np.ones(4)

        # Line 9: Bounds
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        if l_input[0] == "VALUE":
            val = np.array(l_input[1:3])
            bounds = val.astype(np.float)

        elif l_input[0] == "FILE":
            bounds = l_input[1].rstrip()

        else:
            bounds = [-np.inf, np.inf]

        # Line 10: Norms
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        if l_input[0] == "VALUE":
            val = np.array(l_input[1:5])
            lpnorms = val.astype(np.float)

        elif l_input[0] == "FILE":
            lpnorms = l_input[1].rstrip()

        # Line 11: Treshold values
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        if l_input[0] == "VALUE":
            val = np.array(l_input[1:3])
            eps = val.astype(np.float)

        elif l_input[0] == "DEFAULT":
            eps = None

        self.mshfile = mshfile
        self.obsfile = obsfile
        self.topofile = topofile
        self.mstart = mstart
        self._mrefInput = mref
        self._staticInput = staticInput
        self.wgtfile = wgtfile
        self.chi = chi
        self.alphas = alphas
        self.bounds = bounds
        self.lpnorms = lpnorms
        self.eps = eps

    @property
    def mesh(self):
        if getattr(self, "_mesh", None) is None:

            if self.mshfile is not None:
                self._mesh = Mesh.TensorMesh.readUBC(self.basePath + self.mshfile)
            else:
                self._mesh = None

        return self._mesh

    @property
    def topo(self):
        if getattr(self, "_topo", None) is None:

            if getattr(self, "topofile", None) is not None:
                self._topo = np.genfromtxt(self.basePath + self.topofile, skip_header=1)
            else:
                self._topo = None

        return self._topo

    @property
    def survey(self):
        if getattr(self, "_survey", None) is None:
            self._survey = Utils.io_utils.readUBCgravityObservations(
                self.basePath + self.obsfile
            )
        return self._survey

    @property
    def activeCells(self):
        if getattr(self, "_activeCells", None) is None:
            if getattr(self, "topo", None) is not None:
                # Find the active cells
                active = Utils.surface2ind_topo(self.mesh, self.topo, gridLoc="N")

            elif isinstance(self._staticInput, float):
                active = self.m0 != self._staticInput

            else:
                # Read from file active cells with 0:air, 1:dynamic, -1 static
                active = self.activeModel != 0

            # inds = np.asarray(
            #     [
            #         inds for inds, elem in enumerate(active, 1) if elem
            #     ],
            #     dtype=int
            # ) - 1

            self._activeCells = active

            # # Reduce m0 to active space
            # if len(self.m0) > len(self._activeCells):
            #     self._m0 = self.m0[self._activeCells]

        return self._activeCells

    @property
    def staticCells(self):
        if getattr(self, "_staticCells", None) is None:

            # Cells with value 1 in active model are dynamic
            staticCells = self.activeModel[self._activeCells] == -1

            # inds = np.asarray(
            #     [
            #         inds for inds, elem in enumerate(staticCells, 1) if elem
            #     ],
            #     dtype=int
            # ) - 1
            self._staticCells = staticCells

        return self._staticCells

    @property
    def dynamicCells(self):
        if getattr(self, "_dynamicCells", None) is None:

            # Cells with value 1 in active model are dynamic
            dynamicCells = self.activeModel[self._activeCells] == 1

            # inds = np.asarray(
            #     [inds for inds, elem in enumerate(
            #      dynamicCells, 1) if elem], dtype=int
            # ) - 1

            self._dynamicCells = dynamicCells

        return self._dynamicCells

    @property
    def nC(self):
        if getattr(self, "_nC", None) is None:
            self._nC = len(self.activeCells)
        return self._nC

    @property
    def m0(self):
        if getattr(self, "_m0", None) is None:
            if isinstance(self.mstart, float):
                self._m0 = np.ones(self.mesh.nC) * self.mstart
            else:

                self._m0 = Mesh.TensorMesh.readModelUBC(
                    self.mesh, self.basePath + self.mstart
                )

        return self._m0

    @property
    def mref(self):
        if getattr(self, "_mref", None) is None:
            if isinstance(self._mrefInput, float):
                self._mref = np.ones(self.nC) * self._mrefInput
            else:
                self._mref = Mesh.TensorMesh.readModelUBC(
                    self.mesh, self.basePath + self._mrefInput
                )
                # self._mref = self._mref[self.activeCells]
        return self._mref

    @property
    def activeModel(self):
        if getattr(self, "_activeModel", None) is None:
            if isinstance(self._staticInput, str):
                # Read from file active cells with 0:air, 1:dynamic, -1 static
                self._activeModel = Mesh.TensorMesh.readModelUBC(
                    self.mesh, self.basePath + self._staticInput
                )

            else:
                self._activeModel = np.ones(self._mesh.nC)

        return self._activeModel
