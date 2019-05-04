from collections import OrderedDict

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.NDComponents.SynapseModels.BaseSynapseModel import BaseSynapseModel

class Synapse(BaseSynapseModel):
    accesses = ['V']
    updates = ['g'] # conductance (mS/cm^2)
    params = ['weight']

    def __init__(self, params_dict, access_buffers, dt,
                 LPU_id=None, debug=False, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.debug = debug
        self.dt = dt
        self.num_comps = params_dict[self.params[0]].size
        self.dtype = params_dict[self.params[0]].dtype
        self.LPU_id = LPU_id
        self.steps = 1
        self.params_dict = params_dict
        self.access_buffers = access_buffers
        self.ddt = self.dt/self.steps

        self.inputs = {
            k: garray.empty(self.num_comps, dtype=self.access_buffers[k].dtype)
            for k in self.accesses}

        self.retrieve_buffer_funcs = {}
        for k in self.accesses:
            self.retrieve_buffer_funcs[k] = \
                self.get_retrieve_buffer_func(
                    k, dtype=self.access_buffers[k].dtype)

        dtypes = {'dt': self.dtype}
        dtypes.update({k.format(k): self.inputs[
                      k].dtype for k in self.accesses})
        dtypes.update({k: self.params_dict[
                      k].dtype for k in self.params})
        dtypes.update({k: self.dtype for k in self.updates})
        self.update_func = self.get_update_func(dtypes)

    def run_step(self, update_pointers, st=None):
        # retrieve all buffers into a linear array
        for k in self.inputs:
            self.retrieve_buffer(k, st=st)

        self.update_func.prepared_async_call(
            self.update_func.grid, self.update_func.block, st,
            self.num_comps,self.ddt*1000, self.steps,
            *[self.inputs[k].gpudata for k in self.accesses] +
            [self.params_dict[k].gpudata for k in self.params] +
            [update_pointers[k] for k in self.updates])

    def get_update_template(self):
        # The following kernel assumes a maximum of one input connection
        # per neuron

            # this is a kernel that runs 1 step internally for each self.dt
        template = """
__global__ void update(int num_comps, %(dt)s dt,
                       %(V)s* g_V,
                       %(weight)s* g_weight, %(g)s* g_g)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(V)s V;
    %(weight)s weight;

    for(int i = tid; i < num_comps; i += total_threads)
    {
        V = g_V[i];
        weight = g_weight[i];

        g_g[i] = V*weight;
    }
}
"""
       
        return template

    def get_update_func(self, dtypes):
        type_dict = {k: dtype_to_ctype(dtypes[k]) for k in dtypes}
        type_dict.update({'fletter': 'f' if type_dict[self.params[0]] == 'float' else ''})
        mod = SourceModule(self.get_update_template() % type_dict,
                           options=self.compile_options)
        func = mod.get_function("update")
        func.prepare(
            'i' + np.dtype(dtypes['dt']).char + 'i' + 'P' * (len(type_dict) - 2))
        func.block = (256, 1, 1)
        func.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                         (self.num_comps - 1) // 256 + 1), 1)
        return func
