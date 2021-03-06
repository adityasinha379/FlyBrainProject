#!/usr/bin/env python

from itertools import product
import sys

import numpy as np
import h5py
import networkx as nx


def create_lpu_graph(lpu_name, N_driver, N_ring):
    # Set numbers of neurons:
    neu_type = ('driver', 'ring', 'pos', 'rota', 'rotb')
    neu_num = (N_driver, N_ring, N_ring, N_ring, N_ring)

    # Neuron ids are between 0 and the total number of neurons:
    G = nx.MultiDiGraph()

    in_port_idx = 0
    gpot_out_id = 0

    for (t, n) in zip(neu_type, neu_num):
        for i in range(n):
            id = t + "_" + str(i)

            if t == 'driver':
                G.add_node('in_' + str(in_port_idx),
                               **{'class': 'Port',
                                  'name': 'in_' + str(in_port_idx),
                                  'port_type': 'gpot',
                                  'port_io': 'in',
                                  'selector': '/%s/in/gpot/%s' % (lpu_name, in_port_idx)
                                  })

                G.add_node(id,
                           **{'class': 'DrN',
                              'name': id + '_s',
                              'initV': 0.0,
                              'resting_potential': 0.0,
                              'tau': 10.
                              })
                G.add_edge('in_' + str(in_port_idx), id)
                in_port_idx += 1

            # Ring attractor neurons are all attached to output
            # ports (which are represented as separate nodes):
            elif t == 'ring':
                if(gpot_out_id==0):
                    G.add_node(id,
                           **{'class': 'LIN',
                              'name': id + '_s',
                              'initV': 1.,
                              'resting_potential': 0.0,
                              'tau': 6.2
                              })
                else:
                    G.add_node(id,
                               **{'class': 'LIN',
                                  'name': id + '_s',
                                  'initV': 0.,
                                  'resting_potential': 0.0,
                                  'tau': 6.2
                                  })

                G.add_node('out_'+str(gpot_out_id),
                           **{'class': 'Port',
                              'name': 'out_'+str(gpot_out_id),
                              'port_type': 'gpot',
                              'port_io': 'out',
                              'selector': '/%s/out/gpot/%s' % (lpu_name, str(gpot_out_id))
                              })
                G.add_edge(id, 'out_'+str(gpot_out_id))
                gpot_out_id += 1
            
            elif t == 'pos':
                G.add_node('in_' + str(in_port_idx),
                               **{'class': 'Port',
                                  'name': 'in_' + str(in_port_idx),
                                  'port_type': 'gpot',
                                  'port_io': 'in',
                                  'selector': '/%s/in/gpot/%s' % (lpu_name, in_port_idx)
                                  })
                G.add_node(id,
                           **{'class': 'LIN',
                              'name': id + '_s',
                              'initV': 0.,
                              'resting_potential': 0.0,
                              'tau': 10.
                              })
                G.add_edge('in_' + str(in_port_idx), id)
                in_port_idx += 1
                synapse_name = id+'->ring_'+str(i)
                G.add_node(synapse_name,
                               **{'class': 'Synapse',
                                  'name': synapse_name,
                                  'weight': 0.0
                                  })
                G.add_edge(id,synapse_name)
                G.add_edge(synapse_name,
                           'ring_' + str(i))

            # Rota-> +1 clockwise, Rotb-> -1 anticlockwise
            elif t == 'rota':
                G.add_node(id,
                           **{'class': 'RotN',
                              'name': id + '_s',
                              'weight': 0.1
                              })
                G.add_edge('ring_'+str(i),id)
                G.add_edge('driver_0',id)               

            elif t == 'rotb':
                G.add_node(id,
                           **{'class': 'RotN',
                              'name': id + '_s',
                              'weight': 0.1
                              })
                G.add_edge('ring_'+str(i),id)
                G.add_edge('driver_1',id)

    # # Defining Connectivities
    # # Ring Attractor Connections
    for i in range(N_ring):
        # Self Excitation
        fac = 0.9667
        synapse_name = 'Exc_r_'+str(i)+'->'+str(i)
        G.add_node(synapse_name,
                       **{'class': 'Synapse',
                          'name': synapse_name,
                          'weight': 0.6*fac
                          })
        G.add_edge('ring_'+str(i),synapse_name)
        G.add_edge(synapse_name,'ring_'+str(i))           

        for j in (i+1+np.arange(N_ring-1))%N_ring:
          #Inhibitory
            synapse_name = 'Inh_r_'+str(j)+'->'+str(i)
            G.add_node(synapse_name,
                           **{'class': 'Synapse',
                              'name': synapse_name,
                              'weight': -0.1
                              })
            G.add_edge('ring_'+str(j),synapse_name)
            G.add_edge(synapse_name,'ring_'+str(i))
          # Excitatory
            if abs(j-i)<3:
                synapse_name = 'Exc_r_'+str(j)+'->'+str(i)
                G.add_node(synapse_name,
                       **{'class': 'Synapse',
                          'name': synapse_name,
                          'weight': 0.35*fac if abs(j-i) is 1 else 0.225*fac
                          })
                G.add_edge('ring_'+str(j),synapse_name)
                G.add_edge(synapse_name,'ring_'+str(i))
            if abs(j-i)>13:
                synapse_name = 'Exc_r_'+str(j)+'->'+str(i)
                G.add_node(synapse_name,
                       **{'class': 'Synapse',
                          'name': synapse_name,
                          'weight': 0.35*fac if abs(j-i) is 15 else 0.225*fac
                          })
                G.add_edge('ring_'+str(j),synapse_name)
                G.add_edge(synapse_name,'ring_'+str(i))

        # Rota to Ring
        G.add_edge('rota_'+str(i),'ring_'+str((i-1)%N_ring))

        # Rotb to Ring
        G.add_edge('rotb_'+str(i),'ring_'+str((i+1)%N_ring))

    return G


def create_lpu(file_name, lpu_name, N_driver, N_ring):
    g = create_lpu_graph(lpu_name, N_driver, N_ring)
    nx.write_gexf(g, file_name)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lpu_file_name', nargs='?', default='RingAttractorNetwork.gexf.gz',
                        help='LPU file name')
    parser.add_argument('-s', type=int,
                        help='Seed random number generator')
    parser.add_argument('-l', '--lpu', type=str, default='gen',
                        help='LPU name')

    args = parser.parse_args()

    neu_num = [2,16]

    create_lpu(args.lpu_file_name, args.lpu, *neu_num)
    g = nx.read_gexf(args.lpu_file_name)