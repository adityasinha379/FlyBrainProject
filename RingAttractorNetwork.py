#!/usr/bin/env python

from itertools import product
import sys

import numpy as np
import h5py
import networkx as nx


def create_lpu_graph(lpu_name, N_ring, N_driver):
    # Set numbers of neurons:
    neu_type = ('ring', 'pos', 'rota', 'rotb', 'driver')
    neu_num = (N_ring, N_ring, N_ring, N_ring, N_driver)

    # Neuron ids are between 0 and the total number of neurons:
    G = nx.MultiDiGraph()

    in_port_idx = 0
    spk_out_id = 0

    for (t, n) in zip(neu_type, neu_num):
        for i in range(n):
            id = t + "_" + str(i)

            if t == 'driver':
                G.add_node('in_' + str(in_port_idx),
                               **{'class': 'Port',
                                  'name': 'in_' + str(in_port_idx),
                                  'port_type': 'spike',
                                  'port_io': 'in',
                                  'selector': '/%s/in/spk/%s' % (lpu_name, in_port_idx)
                                  })

                G.add_node(id,
                           **{'class': 'LeakyIAF',
                              'name': id + '_s',
                              'initV': np.random.uniform(-60.0, -25.0),
                              'reset_potential': -67.5489770451,
                              'resting_potential': 0.0,
                              'threshold': -25.1355161007,
                              'resistance': 1002.445570216,
                              'capacitance': 0.0669810502993
                              })
                G.add_edge('in_' + str(in_port_idx), id)
                in_port_idx += 1

            # Ring attractor neurons are all attached to output
            # ports (which are represented as separate nodes):
            elif t == 'ring':
                G.add_node(id,
                           **{'class': 'LeakyIAF',
                              'name': id,
                              'initV': np.random.uniform(-60.0, -25.0),
                              'reset_potential': -67.5489770451,
                              'resting_potential': 0.0,
                              'threshold': -25.1355161007,
                              'resistance': 1002.445570216,
                              'capacitance': 0.0669810502993
                              })

                G.add_node('out_'+str(spk_out_id),
                           **{'class': 'Port',
                              'name': 'out_'+str(spk_out_id),
                              'port_type': 'spike',
                              'port_io': 'out',
                              'selector': '/%s/out/spk/%s' % (lpu_name, str(spk_out_id))
                              })
                G.add_edge(id, id + '_port')
                spk_out_id += 1
            
            elif t == 'pos':
                G.add_node('in_' + str(in_port_idx),
                               **{'class': 'Port',
                                  'name': 'in_' + str(in_port_idx),
                                  'port_type': 'spike',
                                  'port_io': 'in',
                                  'selector': '/%s/in/spk/%s' % (lpu_name, in_port_idx)
                                  })
                G.add_node('synapse_in_' + str(in_port_idx) + '_to_ring_' + str(in_port_idx),
                               **{'class': 'GABABSynapse',
                                  'name': 'in_' + str(in_port_idx) + '-' + id,
                                  'gmax': 0.003 * 1e-3,
                                  'a1': 0.09,
                                  'a2': 0.18,
                                  'b1': 0.0012,
                                  'b2': 0.034,
                                  'n': 4,
                                  'gamma':100.0,
                                  'reverse': -95.0                                  
                                  })
                G.add_edge('in_' + str(in_port_idx),
                             'synapse_in_' + str(in_port_idx) + '_to_ring_' + str(in_port_idx))
                G.add_edge('synapse_in_' + str(in_port_idx) + '_to_ring_' + str(in_port_idx),
                           'ring_' + str(in_port_idx))
                in_port_idx += 1

            elif t == 'rota' or t == 'rotb':
                G.add_node(id,
                           **{'class': 'LeakyIAF',
                              'name': id + '_s',
                              'initV': np.random.uniform(-60.0, -25.0),
                              'reset_potential': -67.5489770451,
                              'resting_potential': 0.0,
                              'threshold': -25.1355161007,
                              'resistance': 1002.445570216,
                              'capacitance': 0.0669810502993
                              })

    # Defining Connectivities
    # Ring Attractor Connections
    for i in range(N_ring):
        # Self Excitation
        synapse_name = 'Self_Ex_r_'+'_'+str(i)
        G.add_node(synapse_name,
                       **{'class': 'GABABSynapse',
                          'name': synapse_name,
                          'gmax': 0.003 * 1e-3,
                          'a1': 0.09,
                          'a2': 0.18,
                          'b1': 0.0012,
                          'b2': 0.034,
                          'n': 4,
                          'gamma':100.0,
                          'reverse': -95.0
                          })
        G.add_edge('ring_'+str(i),synapse_name)
        G.add_edge(synapse_name,'ring_'+str(i))           

        for j in rem(i+1+range(N_ring-1),N_ring):
          #Inhibitory
            synapse_name = 'Inhib_r_'+'_'+str(j)+'_'+str(i)
            G.add_node(synapse_name,
                           **{'class': 'GABABSynapse',
                              'name': synapse_name,
                              'gmax': 0.003 * 1e-3,
                              'a1': 0.09,
                              'a2': 0.18,
                              'b1': 0.0012,
                              'b2': 0.034,
                              'n': 4,
                              'gamma':100.0,
                              'reverse': -95.0
                              })
            G.add_edge('ring_'+str(j),synapse_name)
            G.add_edge(synapse_name,'ring_'+str(i))
          # Excitatory
            if abs(j-i)<3:
                synapse_name = 'Excit_r_'+'_'+str(j)+'_'+str(i)
                G.add_node(synapse_name,
                               **{'class': 'GABABSynapse',
                                  'name': synapse_name,
                                  'gmax': 0.003 * 1e-3,
                                  'a1': 0.09,
                                  'a2': 0.18,
                                  'b1': 0.0012,
                                  'b2': 0.034,
                                  'n': 4,
                                  'gamma':100.0,
                                  'reverse': -95.0,
                                  })
                G.add_edge('ring_'+str(j),synapse_name)
                G.add_edge(synapse_name,'ring_'+str(i))

    # Rota-> +1 clockwise, Rotb-> -1 anticlockwise
    for i in range(N_ring):
    # Rota
      # Ring to Rota
        synapse_name = 'ring_'+str(i)+'_rota_'+str(i)
        G.add_node(synapse_name,
                       **{'class': 'GABABSynapse',
                          'name': synapse_name,
                          'gmax': 0.003 * 1e-3,
                          'a1': 0.09,
                          'a2': 0.18,
                          'b1': 0.0012,
                          'b2': 0.034,
                          'n': 4,
                          'gamma':100.0,
                          'reverse': -95.0,
                          })
        G.add_edge('ring_'+str(j),synapse_name)
        G.add_edge(synapse_name,'rota_'+str(i))

      # Rota to Ring
        synapse_name = 'rota_'+str(i)+'_ring_'+str(i+1)
        G.add_node(synapse_name,
                       **{'class': 'GABABSynapse',
                          'name': synapse_name,
                          'gmax': 0.003 * 1e-3,
                          'a1': 0.09,
                          'a2': 0.18,
                          'b1': 0.0012,
                          'b2': 0.034,
                          'n': 4,
                          'gamma':100.0,
                          'reverse': -95.0
                          }) 
        G.add_edge('rota_'+str(j),synapse_name)
        G.add_edge(synapse_name,'ring_'+str(i+1))

        # Driver to Rota
        synapse_name = 'driver_0_rota_'+str(i)
        G.add_node(synapse_name,
                       **{'class': 'GABABSynapse',
                          'name': synapse_name,
                          'gmax': 0.003 * 1e-3,
                          'a1': 0.09,
                          'a2': 0.18,
                          'b1': 0.0012,
                          'b2': 0.034,
                          'n': 4,
                          'gamma':100.0,
                          'reverse': -95.0
                          })
        G.add_edge('driver_0',synapse_name)
        G.add_edge(synapse_name,'rota_'+str(i))

      # Rota to Driver
        synapse_name = 'rota_'+str(i)+'driver_0'
        G.add_node(synapse_name,
                       **{'class': 'GABABSynapse',
                          'name': synapse_name,
                          'gmax': 0.003 * 1e-3,
                          'a1': 0.09,
                          'a2': 0.18,
                          'b1': 0.0012,
                          'b2': 0.034,
                          'n': 4,
                          'gamma':100.0,
                          'reverse': -95.0
                          })
        G.add_edge('rota_'+str(i),'driver_0')
        G.add_edge('rota_'+str(i),synapse_name)
        G.add_edge(synapse_name,'driver_0')     

    # Rotb
      # Ring to Rotb
        synapse_name = 'ring_'+str(i)+'_rotb_'+str(i)
        G.add_node(synapse_name,
                       **{'class': 'GABABSynapse',
                          'name': synapse_name,
                          'gmax': 0.003 * 1e-3,
                          'a1': 0.09,
                          'a2': 0.18,
                          'b1': 0.0012,
                          'b2': 0.034,
                          'n': 4,
                          'gamma':100.0,
                          'reverse': -95.0
                          })
        G.add_edge('ring_'+str(j),synapse_name)
        G.add_edge(synapse_name,'rotb_'+str(i))

      # Rotb to Ring
        synapse_name = 'rotb_'+str(i)+'_ring_'+str(i-1)
        G.add_node(synapse_name,
                       **{'class': 'GABABSynapse',
                          'name': synapse_name,
                          'gmax': 0.003 * 1e-3,
                          'a1': 0.09,
                          'a2': 0.18,
                          'b1': 0.0012,
                          'b2': 0.034,
                          'n': 4,
                          'gamma':100.0,
                          'reverse': -95.0
                          })
        G.add_edge('rotb_'+str(j),synapse_name)
        G.add_edge(synapse_name,'ring_'+str(i-1))

        # Driver to Rotb
        synapse_name = 'driver_1_rotb_'+str(i)
        G.add_node(synapse_name,
                       **{'class': 'GABABSynapse',
                          'name': synapse_name,
                          'gmax': 0.003 * 1e-3,
                          'a1': 0.09,
                          'a2': 0.18,
                          'b1': 0.0012,
                          'b2': 0.034,
                          'n': 4,
                          'gamma':100.0,
                          'reverse': -95.0
                          })
        G.add_edge('driver_1'+str(j),synapse_name)
        G.add_edge(synapse_name,'rotb_'+str(i))

      # Rota to Driver
        synapse_name = 'rotb_'+str(i)+'driver_1'
        G.add_node(synapse_name,
                       **{'class': 'GABABSynapse',
                          'name': synapse_name,
                          'gmax': 0.003 * 1e-3,
                          'a1': 0.09,
                          'a2': 0.18,
                          'b1': 0.0012,
                          'b2': 0.034,
                          'n': 4,
                          'gamma':100.0,
                          'reverse': -95.0
                          })
        G.add_edge('rotb_'+str(i),synapse_name)
        G.add_edge(synapse_name,'driver_1')        

    return G


def create_lpu(file_name, lpu_name, N_sensory, N_local, N_proj):
    """
    Create a generic LPU graph.

    Creates a GEXF file containing the neuron and synapse parameters for an LPU
    containing the specified number of local and projection neurons. The GEXF
    file also contains the parameters for a set of sensory neurons that accept
    external input. All neurons are either spiking or graded potential neurons;
    the Leaky Integrate-and-Fire model is used for the former, while the
    Morris-Lecar model is used for the latter (i.e., the neuron's membrane
    potential is deemed to be its output rather than the time when it emits an
    action potential). Synapses use either the alpha function model or a
    conductance-based model.

    Parameters
    ----------
    file_name : str
        Output GEXF file name.
    lpu_name : str
        Name of LPU. Used in port identifiers.
    N_sensory : int
        Number of sensory neurons.
    N_local : int
        Number of local neurons.
    N_proj : int
        Number of project neurons.

    Returns
    -------
    g : networkx.MultiDiGraph
        Generated graph.
    """

    g = create_lpu_graph(lpu_name, N_sensory, N_local, N_proj)
    nx.write_gexf(g, file_name)


def create_input(file_name, N_sensory, dt=1e-4, dur=1.0, start=0.3, stop=0.6, I_max=0.6):
    """
    Create input stimulus for sensory neurons in artificial LPU.

    Creates an HDF5 file containing input signals for the specified number of
    neurons. The signals consist of a rectangular pulse of specified duration
    and magnitude.

    Parameters
    ----------
    file_name : str
        Name of output HDF5 file.
    g: networkx.MultiDiGraph
        NetworkX graph object representing the LPU
    dt : float
        Time resolution of generated signal.
    dur : float
        Duration of generated signal.
    start : float
        Start time of signal pulse.
    stop : float
        Stop time of signal pulse.
    I_max : float
        Pulse magnitude.
    """

    Nt = int(dur / dt)
    t = np.arange(0, dt * Nt, dt)

    uids = ["sensory_" + str(i) for i in range(N_sensory)]

    uids = np.array(uids, dtype = 'S')

    I = np.zeros((Nt, N_sensory), dtype=np.float64)
    I[np.logical_and(t > start, t < stop)] = I_max

    with h5py.File(file_name, 'w') as f:
        f.create_dataset('I/uids', data=uids)
        f.create_dataset('I/data', (Nt, N_sensory),
                         dtype=np.float64,
                         data=I)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lpu_file_name', nargs='?', default='generic_lpu.gexf.gz',
                        help='LPU file name')
    parser.add_argument('in_file_name', nargs='?', default='generic_input.h5',
                        help='Input file name')
    parser.add_argument('-s', type=int,
                        help='Seed random number generator')
    parser.add_argument('-l', '--lpu', type=str, default='gen',
                        help='LPU name')

    args = parser.parse_args()

    if args.s is not None:
        np.random.seed(args.s)
    dt = 1e-4
    dur = 1.0
    start = 0.3
    stop = 0.6
    I_max = 0.6
    neu_num = [np.random.randint(31, 40) for i in range(3)]

    create_lpu(args.lpu_file_name, args.lpu, *neu_num)
    g = nx.read_gexf(args.lpu_file_name)
    create_input(args.in_file_name, neu_num[0], dt, dur, start, stop, I_max)
    create_lpu(args.lpu_file_name, args.lpu, *neu_num)
