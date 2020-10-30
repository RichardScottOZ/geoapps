from . import Utils
from . import Props
from . import DataMisfit
from . import Regularization
from . import ObjectiveFunction
from . import Optimization
from time import time
import properties
import numpy as np
from scipy.sparse import csr_matrix as csr
import gc
import dask
import dask.array as da


class BaseInvProblem(Props.BaseSimPEG):
    """BaseInvProblem(dmisfit, reg, opt)"""

    #: Trade-off parameter
    beta = 1.0

    #: Print debugging information
    debug = False

    #: Set this to a SimPEG.Utils.Counter() if you want to count things
    counter = None

    #: DataMisfit
    dmisfit = None

    #: Regularization
    reg = None

    #: Optimization program
    opt = None

    #: Use BFGS
    bfgs = True

    #: List of strings, e.g. ['_MeSigma', '_MeSigmaI']
    deleteTheseOnModelUpdate = []

    model = Props.Model("Inversion model.")
    _client = None

    @properties.observer("model")
    def _on_model_update(self, value):
        """
            Sets the current model, and removes dependent properties
        """
        for prop in self.deleteTheseOnModelUpdate:
            if hasattr(self, prop):
                delattr(self, prop)

    def __init__(self, dmisfit, reg, opt, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(dmisfit, DataMisfit.BaseDataMisfit) or isinstance(
            dmisfit, ObjectiveFunction.BaseObjectiveFunction
        ), "dmisfit must be a DataMisfit or ObjectiveFunction class."
        assert isinstance(reg, Regularization.BaseRegularization) or isinstance(
            reg, ObjectiveFunction.BaseObjectiveFunction
        ), "reg must be a Regularization or Objective Function class."
        self.dmisfit = dmisfit
        self.reg = reg
        self.opt = opt
        # TODO: Remove: (and make iteration printers better!)
        self.opt.parent = self
        self.reg.parent = self
        self.dmisfit.parent = self

    @Utils.callHooks("startup")
    def startup(self, m0):
        """startup(m0)

            Called when inversion is first starting.
        """
        if self.debug:
            print("Calling InvProblem.startup")

        if hasattr(self.reg, "mref") and getattr(self.reg, "mref", None) is None:
            print("SimPEG.InvProblem will set Regularization.mref to m0.")
            self.reg.mref = m0

        if isinstance(
            self.reg, ObjectiveFunction.ComboObjectiveFunction
        ) and not isinstance(self.reg, Regularization.BaseComboRegularization):
            for fct in self.reg.objfcts:
                if hasattr(fct, "mref") and getattr(fct, "mref", None) is None:
                    print("SimPEG.InvProblem will set Regularization.mref to m0.")
                    fct.mref = m0

        self.phi_d = np.nan
        self.phi_m = np.nan

        self.model = m0

        # if self.bfgs:
        #     if isinstance(self.dmisfit, DataMisfit.BaseDataMisfit):
        #         print("""SimPEG.InvProblem is setting bfgsH0 to the inverse of the eval2Deriv.
        #               ***Done using same Solver and solverOpts as the problem***"""
        #               )
        #         self.opt.bfgsH0 = self.dmisfit.prob.Solver(
        #             self.reg.deriv2(self.model), **self.dmisfit.prob.solverOpts
        #                                                    )
        #     elif isinstance(self.dmisfit,
        #                     ObjectiveFunction.BaseObjectiveFunction
        #                     ):
        #         for objfct in self.dmisfit.objfcts:
        #             if isinstance(objfct, DataMisfit.BaseDataMisfit):
        #                 print("""SimPEG.InvProblem is setting bfgsH0 to the inverse of the eval2Deriv.
        #                       ***Done using same Solver and solverOpts as the {} problem***""".format(
        #                         objfct.prob.__class__.__name__
        #                     )
        #                 )
        #                 self.opt.bfgsH0 = objfct.prob.Solver(
        #                     self.reg.deriv2(self.model), **objfct.prob.solverOpts
        #                 )
        #                 break

    @property
    def warmstart(self):
        return getattr(self, "_warmstart", [])

    @warmstart.setter
    def warmstart(self, value):
        assert type(value) is list, "warmstart must be a list."
        for v in value:
            assert type(v) is tuple, "warmstart must be a list of tuples (m, u)."
            assert (
                len(v) == 2
            ), "warmstart must be a list of tuples (m, u). YOURS IS NOT LENGTH 2!"
            assert isinstance(
                v[0], np.ndarray
            ), "first warmstart value must be a model."
        self._warmstart = value

    @property
    def client(self):
        # if getattr(self, '_client', None) is None:
        #     self._client = Client(processes=False)

        return self._client

    @client.setter
    def client(self, client):
        self._client = client

    def getFields(self, m, store=False, deleteWarmstart=True):
        f = None

        for mtest, u_ofmtest in self.warmstart:
            if m is mtest:
                f = u_ofmtest
                if self.debug:
                    print("InvProb is Warm Starting!")
                break

        if f is None:
            if isinstance(self.dmisfit, DataMisfit.BaseDataMisfit):
                f = self.dmisfit.prob.fields(m)
            elif isinstance(self.dmisfit, ObjectiveFunction.BaseObjectiveFunction):
                f = []
                for objfct in self.dmisfit.objfcts:
                    if hasattr(objfct, "prob"):
                        f += [objfct.prob.fields(m)]
                    else:
                        f += []

        if deleteWarmstart:
            self.warmstart = []
        if store:
            self.warmstart += [(m, f)]

        return f

    def get_dpred(self, m, f):

        if isinstance(self.dmisfit, DataMisfit.BaseDataMisfit):
            return self.dmisfit.survey.dpred(m, f=f)
        elif isinstance(self.dmisfit, ObjectiveFunction.BaseObjectiveFunction):
            dpred = []
            index = []
            for i, objfct in enumerate(self.dmisfit.objfcts):
                if hasattr(objfct, "survey"):
                    dpred += [objfct.survey.dpred(m, f=f[i])]
                    index += [np.where(objfct.survey.ind)]
                else:
                    dpred += []
                    index += []

            dpred = self.client.submit(
                da.compute, self.client.scatter(da.hstack(dpred))
            ).result()[0]
            index = np.hstack(index)

            return dpred[index]

    @Utils.timeIt
    def evalFunction(self, m, return_g=True, return_H=True):
        """evalFunction(m, return_g=True, return_H=True)
        """

        self.model = m
        gc.collect()

        # Store fields if doing a line-search
        f = self.getFields(m, store=(return_g is False and return_H is False))

        m_future = self.client.scatter(m, broadcast=True)

        # if isinstance(self.dmisfit, DataMisfit.BaseDataMisfit):
        phi_d = np.asarray(self.dmisfit(m, f=f).result())
        # self.dpred = self.get_dpred(m, f=f)

        # phi_d = np.linalg.norm(self.dmisfit.W * self.dpred)
        reg = self.reg(m)

        if isinstance(reg, dask.distributed.Future):
            reg = reg.result()
        phi_m = np.asarray(reg)

        self.phi_d, self.phi_d_last = phi_d, self.phi_d
        self.phi_m, self.phi_m_last = phi_m, self.phi_m

        phi = phi_d + self.beta * phi_m

        out = (phi,)
        if return_g:
            phi_dDeriv = self.dmisfit.deriv(m, f=f)
            phi_mDeriv = self.reg.deriv(m)

            g = phi_dDeriv.result() + self.beta * phi_mDeriv
            out += (g,)

        if return_H:

            # if isinstance(v, dask.distributed.Future):
            #     v = v.result()
            #
            # phi_d2Deriv = self.dmisfit.deriv2(m, v, f=f)
            # if isinstance(phi_d2Deriv, dask.distributed.Future):
            #     phi_m2Deriv = self.beta * reg_deriv2 * np.asarray(v)

            def H_fun(v):
                # phi_d2Deriv = self.dmisfit.deriv2(m, v, f=f)
                # if isinstance(phi_d2Deriv, dask.array.Array):
                #     # future = self.client.scatter(self.beta * reg_deriv2)
                #     # dmudm_v = self.client.submit(dask.delayed(csr.dot), future, v)
                #     # phi_m2Deriv = self.client.scatter(da.from_delayed(
                #     #     dmudm_v, dtype=float, shape=[m.shape[0]])
                #     # )
                #     # return self.client.submit(da.add, phi_d2Deriv, phi_m2Deriv).result()
                #     dmudm_v = dask.delayed(csr.dot)(reg_deriv2, v)
                #     phi_m2Deriv = da.from_delayed(dmudm_v, dtype=float, shape=[m.shape[0]])
                #     return phi_d2Deriv + self.beta * phi_m2Deriv
                #
                # else:
                v_future = self.client.scatter(v, broadcast=True)

                # tc = time()
                phi_m2Deriv = self.reg.deriv2(m, v=v)
                # print(f"Reg {time()-tc}")
                #
                # tc = time()
                phi_d2Deriv = self.dmisfit.deriv2(m, v, f=f)
                # print(f"Misfit {time() - tc}")

                H = phi_d2Deriv.result() + self.beta * phi_m2Deriv
                return H

            H = H_fun  # sp.linalg.LinearOperator((m.size, m.size), H_fun, dtype=m.dtype)
            out += (H,)
        return out if len(out) > 1 else out[0]
