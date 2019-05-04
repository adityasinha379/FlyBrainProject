#!/usr/bin/env python

"""
Visualize generic LPU demo output.
Notes
-----
Generate demo output by running
python generic_demo.py
"""

import numpy as np
import matplotlib as mpl
mpl.use('agg')

import h5py
import neurokernel.LPU.utils.visualizer as vis
import networkx as nx

nx.readwrite.gexf.GEXF.convert_bool = {'false':False, 'False':False,
                                       'true':True, 'True':True}

G = nx.read_gexf('./RingAttractorNetwork.gexf.gz')
neu_out = sorted([k for k, n in G.node.items() if \
                   n['name'][:4] == 'ring' and \
                   n['class'] == 'LIN'])

N_driver = 2
N_ring = 16

in_uid = ["driver_" + str(i) for i in range(N_driver)]
in_uid.extend(["ring_" + str(i) for i in range(N_ring)])
print(in_uid[:2])

V = vis.visualizer()
V.add_LPU('./input.h5', LPU='RingAttractorIn', is_input=True)
V.add_plot({'type':'waveform', 'uids': [in_uid[:2]], 'variable':'V'},
           'AVDU+Pos Inputs')

V.add_LPU('./output.h5',  'RingAttractorOut',
          gexf_file='./RingAttractorNetwork.gexf.gz')
V.add_plot({'type':'waveform', 'uids': [neu_out[:2]], 'variable': 'V','title': 'Output'},
            'Ring Outputs')

V.rows = 2
V.cols = 1
V.fontsize = 8
V.xlim = [0, 1.0]

gen_video = True
if gen_video:
    V.out_filename = 'output.mp4'
    V.codec = 'mpeg4'
    V.run()
else:
    V.update_interval = None
    V.run('output.png')