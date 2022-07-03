'''
XML Parser for sbs files, providing empirical analysis on graph structures.
'''
import os
import sys
import argparse
import xml.etree.ElementTree as ET

from diffmat.sbs_converter.sbs_parser import type_dict

# Nodes list
nodes_lib = type_dict.keys()

def sbs_examine(file_name, buffer_count=3):
    """Examine a *.sbs file to find unimplemented nodes.

    Args:
        file_name (str): source file name
        buffer_count (int, optional): maximum number of unimplemented nodes allowed. Defaults to 3.

    Returns:
        Bool: whether the material contains too many unimplemented nodes
        Set: the set of unimplemented nodes
    """
    # build xml tree
    tree = ET.parse(file_name)
    root = tree.getroot()

    # scan dependency
    # determine if user-specific sbs is used
    outside_deps_set = set()
    for dep in root.iter('dependency'):
        dep_file_name = dep.find('filename').get('v')
        if dep_file_name[:6] != 'sbs://':
            outside_deps_set.add(dep_file_name)

    # determine the name of 'connexions' or 'connections'
    connection_name = 'connections'
    for i in root.iter('connexions'):
        connection_name = 'connexions'
        break

    # scan compNodes
    unimplemented_nodes_list = set()
    tile_related_nodes_list = set()
    for node in root.iter('compNode'):
        # check if the node has connections
        # if no connection, consider it as generator
        if not node.find(connection_name):
            pass
        else:
            node_imp = node.find('compImplementation')[0]
            # input node
            if node_imp.tag == 'compInputBridge':
                node_name = None
            # output node
            elif node_imp.tag == 'compOutputBridge':
                node_name = None
            # non-atomic node
            elif node_imp.tag == 'compInstance':
                path = node_imp.find('path')
                if path.get('v') is None:
                    path = path.find('value').get('v')
                else:
                    path = path.get('v')
                node_name = path[path.rfind('/') + 1: path.rfind('?')]
            # atomic node
            elif node_imp.tag == 'compFilter':
                node_name = node_imp.find('filter').get('v')
            else:
                raise NotImplementedError('Unrecognized node type: {}'.format(node_imp.tag))
            
            # process the found node
            if node_name is not None:
                if node_name in nodes_lib:
                    pass
                else:
                    unimplemented_nodes_list.add(node_name)

    if unimplemented_nodes_list == set() or len(unimplemented_nodes_list) <= buffer_count:
        return True, unimplemented_nodes_list
    else:
        return False, unimplemented_nodes_list
    
if __name__ == '__main__':
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Automatic examiner of SBS documents to PyTorch modules for optimization.')
    parser.add_argument('input_file', metavar='FILE', help='Path to input *.sbs file.')
    parser.add_argument('-b', '--buffer-count', default=3, type=int, help='Allowed number of unimplemented nodes')

    # Parse arguments
    namespace = parser.parse_args()
    input_file_path = namespace.input_file
    buffer_count = namespace.buffer_count

    # examine
    success_flag, missing_list = sbs_examine(input_file_path, buffer_count)
    print('success_flag: ', success_flag)
    print('missing_list: :', missing_list)