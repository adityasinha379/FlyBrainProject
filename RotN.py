
from collections import OrderedDict

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.NDComponents.SynapseModels.BaseSynapseModel import BaseSynapseModel

class RotN(BaseSynapseModel):
    accesses = ['V1','V2']
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
        self.nsteps = 1
        self.params_dict = params_dict
        self.access_buffers = access_buffers
        self.ddt = self.dt/self.nsteps

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
            self.num_comps, self.ddt*1000, self.nsteps,
            *[self.inputs[k].gpudata for k in self.accesses] +
            [self.params_dict[k].gpudata for k in self.params] +
            [update_pointers[k] for k in self.updates])

    def get_update_template(self):
        # The following kernel assumes a maximum of one input connection
        # per neuron

            # this is a kernel that runs 1 step internally for each self.dt
        template = """
__global__ void update(int num_comps, %(dt)s dt, int steps,
                       %(V1)s* g_V1, %(V2)s* g_V2,
                       %(weight)s* g_weight, %(g)s* g_g)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(V1)s V1;
    %(V2)s V2;
    %(weight)s weight;

    for(int i = tid; i < num_comps; i += total_threads)
    {
        V1 = g_V1[i];
        V2 = g_V2[i];
        weight = g_weight[i];

        g_g[i] = V1*V2*weight;
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

# testing function
if __name__ == '__main__':
    import argparse
    import itertools

    import networkx as nx
    import h5py

    from neurokernel.tools.logging import setup_logger
    import neurokernel.core_gpu as core
    from neurokernel.LPU.LPU import LPU
    from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
    from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
    from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
    import neurokernel.mpi_relaunch

    dt = 1e-4
    dur = 1.0
    steps = int(dur / dt)

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False,
                        dest='debug', action='store_true',
                        help='Write connectivity structures and inter-LPU routed data in debug folder')
    parser.add_argument('-l', '--log', default='none', type=str,
                        help='Log output to screen [file, screen, both, or none; default:none]')
    parser.add_argument('-s', '--steps', default=steps, type=int,
                        help='Number of steps [default: %s]' % steps)
    parser.add_argument('-g', '--gpu_dev', default=0, type=int,
                        help='GPU device number [default: 0]')
    args = parser.parse_args()

    file_name = None
    screen = False
    if args.log.lower() in ['file', 'both']:
        file_name = 'neurokernel.log'
    if args.log.lower() in ['screen', 'both']:
        screen = True
    logger = setup_logger(file_name=file_name, screen=screen)

    man = core.Manager()

    G = nx.MultiDiGraph()

    G.add_node('synapse0', **{
               'class': 'Synapse',
               'name': 'Synapse',
               'weight': 1.
               })

    comp_dict, conns = LPU.graph_to_dicts(G)

    fl_input_processor_1 = StepInputProcessor('V1', ['synapse0'], 6.0, 0.1, 0.9)
    fl_input_processor_2 = StepInputProcessor('V2', ['synapse0'], 2.0, 0.2, 0.8)
    fl_output_processor = FileOutputProcessor(
        [('g', None)], 'new_output.h5', sample_interval=1)

    man.add(LPU, 'syn', dt, comp_dict, conns,
            device=args.gpu_dev, input_processors=[fl_input_processor_1,fl_input_processor_2],
            output_processors=[fl_output_processor], debug=args.debug)

    man.spawn()
    man.start(steps=args.steps)
    man.wait()