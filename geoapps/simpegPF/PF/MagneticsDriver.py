import re
import os
from .. import Utils
import numpy as np
from ..Utils import mkvc
from . import Magnetics
from discretize import TensorMesh


class MagneticsDriver_Inv:
    """docstring for MagneticsDriver_Inv"""

    def __init__(self, input_file=None):
        if input_file is not None:
            self.basePath = os.path.sep.join(input_file.split(os.path.sep)[:-1])
            if len(self.basePath) > 0:
                self.basePath += os.path.sep
            self.readDriverFile(input_file.split(os.path.sep)[-1])

    def readDriverFile(self, input_file):
        """
            Read input files for forward modeling MAG data with integral form
            INPUT:
            input_file: File name containing the forward parameter

            OUTPUT:
            mshfile
            obsfile
            topofile
            start model
            ref model
            mag model
            weightfile
            chi_target
            as, ax ,ay, az
            upper, lower bounds
            lp, lqx, lqy, lqz

            # All files should be in the working directory,
            # otherwise the path must be specified.

        """

        fid = open(self.basePath + input_file)

        # Line 1: Mesh
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        mshfile = l_input[1].rstrip()

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

        # Line 6: Magnetization model
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        if l_input[0] == "DEFAULT":
            magfile = None

        elif l_input[0] == "FILE":
            magfile = l_input[1].rstrip()

        # Line 7: Cell weights
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        if l_input[0] == "DEFAULT":
            wgtfile = []

        elif l_input[0] == "FILE":
            wgtfile = l_input[1].rstrip()

        # Line 8: Target chi-factor
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        if l_input[0] == "DEFAULT":
            chi = 1.0

        elif l_input[0] == "VALUE":
            chi = float(l_input[1])

        # Line 9: Alpha values
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        if l_input[0] == "VALUE":

            val = np.array(l_input[1:5])
            alphas = val.astype(np.float)

        elif l_input[0] == "DEFAULT":

            alphas = np.ones(4)

        # Line 10: Bounds
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        if l_input[0] == "VALUE":
            val = np.array(l_input[1:3])
            bounds = val.astype(np.float)

        elif l_input[0] == "FILE":
            bounds = l_input[1].rstrip()

        else:
            bounds = [-np.inf, np.inf]

        # Line 11: Norms
        line = fid.readline()
        l_input = re.split(r"[!\s]", line)
        if l_input[0] == "VALUE":
            val = np.array(l_input[1:13])
            lpnorms = val.astype(np.float)

        elif l_input[0] == "FILE":
            lpnorms = l_input[1].rstrip()

        # Line 12: Treshold values
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
        self.magfile = magfile
        self.wgtfile = wgtfile
        self.chi = chi
        self.alphas = alphas
        self.bounds = bounds
        self.lpnorms = lpnorms
        self.eps = eps

    @property
    def mesh(self):
        if getattr(self, "_mesh", None) is None:
            self._mesh = TensorMesh.readUBC(self.basePath + self.mshfile)
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
            self._survey, _ = Utils.io_utils.readUBCmagneticsObservations(
                self.basePath + self.obsfile
            )
        return self._survey

    @property
    def activeCells(self):
        if getattr(self, "_activeCells", None) is None:
            if getattr(self, "topo", None) is not None:
                # Find the active cells
                active = Utils.surface2ind_topo(self.mesh, self.topo, "N")

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
            staticCells = self.activeModel[self.activeCells] == -1

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
            dynamicCells = self.activeModel[self.activeCells] == 1

            # inds = np.asarray(
            #     [
            #         inds for inds, elem in enumerate(dynamicCells, 1) if elem
            #     ],
            #     dtype=int
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
                self._m0 = np.ones(self.nC) * self.mstart
            else:
                self._m0 = TensorMesh.readModelUBC(
                    self.mesh, self.basePath + self.mstart
                )

        return self._m0

    @property
    def mref(self):
        if getattr(self, "_mref", None) is None:
            if isinstance(self._mrefInput, float):
                self._mref = np.ones(self.nC) * self._mrefInput
            else:
                self._mref = TensorMesh.readModelUBC(
                    self.mesh, self.basePath + self._mrefInput
                )

                # # Reduce to active space
                # self._mref = self._mref[self.activeCells]

        return self._mref

    @property
    def activeModel(self):
        if getattr(self, "_activeModel", None) is None:
            if self._staticInput == "FILE":
                # Read from file active cells with 0:air, 1:dynamic, -1 static
                self._activeModel = TensorMesh.readModelUBC(
                    self.mesh, self.basePath + self._staticInput
                )

            else:
                self._activeModel = np.ones(self._mesh.nC)

        return self._activeModel

    @property
    def magnetizationModel(self):
        """
            magnetization vector
        """

        if getattr(self, "magfile", None) is None:

            M = Magnetics.dipazm_2_xyz(
                np.ones(self.nC) * self.survey.srcField.param[1],
                np.ones(self.nC) * self.survey.srcField.param[2],
            )

        else:

            with open(self.basePath + self.magfile) as f:
                magmodel = f.read()

            magmodel = magmodel.splitlines()
            M = []

            for line in magmodel:
                M.append(map(float, line.split()))

            # Convert list to 2d array
            M = np.vstack(M)

            # Cycle through three components and permute from UBC to SimPEG
            for ii in range(3):
                m = np.reshape(
                    M[:, ii], (self.mesh.nCz, self.mesh.nCx, self.mesh.nCy), order="F"
                )

                m = m[::-1, :, :]
                m = np.transpose(m, (1, 2, 0))
                M[:, ii] = Utils.mkvc(m)

        self._M = M

        return self._M


def actIndFull2layer(mesh, actInd):
    """
    Function to extract upper layer (topo) of an
    active index vector
    """

    # Convert the actind to bool
    if not isinstance(actInd, bool):

        actIndFull = np.zeros(mesh.nC, dtype=bool)
        actIndFull[actInd] = True

    actIndFull = actIndFull.reshape(mesh.vnC, order="F")

    actIndLayer = np.zeros(mesh.vnC, dtype=bool)
    for ii in range(mesh.nCx):
        for jj in range(mesh.nCy):

            zcol = actIndFull[ii, jj, :]
            actIndLayer[ii, jj, np.where(zcol)[0][-1]] = True

    return np.asarray(np.where(mkvc(actIndLayer))[0], dtype=int)
