

"""
@yizzhan
yizzhan added validation script
Latest commit 321ebc5 21 hours ago
 History
 1 contributor
316 lines (264 sloc)  11.6 KB
"""

import os
import glob
import numpy as np
import xml
import xml.dom.minidom
from xml.etree.ElementTree import Element, SubElement
from pydoc import locate


def listSubDirectory(path, join_path=False):
    '''
    List subdirectories of path.
    :param path:        the path to query
    :param join_path:   True, return absolute path of each subdirectory.  False, return basename of the subdirectory, without path
    :return:            a list of dirs
    '''
    if join_path:
        return [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
    else:
        return [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]


def getAllClipsIdx(scan_path):
    '''
    Get index of clips of a scan.
    If data is arranged in the following format:
    -00_rgbd
        -clip0
        -clip1
    -01_rgbd
        -clip1
        -clip2
    -02_rgbd
        -clip1
        -clip4
        -clipX
    the return will be [0, 1, 2, 4]. clipX will be ignored since it is not numbered
    :param scan_path:   Path of the scan
    :return:            list of numerical index of clips
    '''
    cam_paths = glob.glob(('%s\\*_rgbd' % scan_path))
    all_clip_idx = []

    # for each camera inside this scan, add clip_idx of this camera into all_clip_idx
    for cam_path in cam_paths:
        clip_paths = glob.glob(('%s\\clip*' % cam_path))
        clip_dirs = [os.path.basename(x) for x in clip_paths]
        clip_idxs = [int(x[4:]) for x in clip_dirs if x[4:].isdigit()]
        all_clip_idx += clip_idxs

    all_clip_idx = list(set(all_clip_idx))  # remove duplicate
    all_clip_idx.sort()

    return all_clip_idx


class ClassBase(object):
    '''
    A base class with XML I/O support
    This class base is able to convert all member variables into a XML node, or parse a XML node to init all member variables.
    If you want to save and load data with XML automatically, all you need to do is inherent this class
    '''

    def __init__(self):
        pass

    def create_xml_node(self, root_name=''):
        # create root node
        if root_name == '':
            root = Element(self.__class__.__name__)
        else:
            root = Element(root_name)

        # get all member variables
        member_dict = vars(self)
        member_name = list(vars(self).keys())

        # for all member variables, create sub element to root
        for member in member_name:
            # skip None type
            if type(member_dict[member]) is None:
                continue

            node = Element(member)
            node.set('type', type(member_dict[member]).__name__)

            # if the member is numpy array
            if type(member_dict[member]) is np.ndarray:
                # skip empty
                size = member_dict[member].size
                if size == 0:
                    continue

                # create attribute
                np_shape_str = ''
                np_shape = member_dict[member].shape
                for i in np_shape:
                    np_shape_str = np_shape_str + str(i) + ' '
                node.set('shape', np_shape_str)
                node.set('dtype', str(member_dict[member].dtype))

                # create linear string
                node.text = ''
                flatten = member_dict[member].reshape(size)
                for i in range(size):
                    node.text += str(flatten[i])
                    node.text += ' '

            # if the member is a list, assume all data in list is basic type
            elif type(member_dict[member]) is list or type(member_dict[member]) is tuple:
                # skip empty list
                if len(member_dict[member]) == 0:
                    continue

                # create attribute
                node.set('len', str(len(member_dict[member])))

                # create child node for each element
                for i in range(len(member_dict[member])):
                    child = SubElement(node, 'elem')
                    child.set('index', str(i))
                    child.set('type', type(member_dict[member][i]).__name__)
                    if member_dict[member][i] is not None:
                        child.text = str(member_dict[member][i])

            # if the member is basic python type
            else:
                node.text = str(member_dict[member])
                if node.text == '':
                    continue

            # create the sub node
            root.append(node)

        # return the root (None if empty)
        if len(list(root)) > 0:
            return root
        else:
            return None

    def parse_xml_node(self, root):
        member_dict = vars(self)
        member_name = list(vars(self).keys())

        for member in member_name:
            children = root.findall(member)
            if len(children) == 0:
                continue

            # use the first child
            if len(children) != 1:
                print('Warning: ParseXmlNode (%s), %d nodes found, just read the first one' % (member, len(children)))
            child = children[0]

            # validate type
            attrib = child.attrib
            if 'type' in attrib.keys():
                if type(member_dict[member]).__name__ != attrib['type']:
                    print(
                        'Warning: ParseXmlNode (%s), type not match, use default type, default type (%s), xml type (%s)' % (
                            member, type(member_dict[member]).__name__, attrib['type']))

            # the node is numpy array
            if type(member_dict[member]) is np.ndarray:
                # validate dtype
                if 'dtype' in attrib.keys():
                    if str(member_dict[member].dtype) != attrib['dtype']:
                        print('Warning: ParseXmlNode (%s), dtype not match, class dtype (%s), xml dtype (%s)' % (
                            member, str(member_dict[member].dtype), attrib['dtype']))

                # create flatten array
                np_array_str = child.text.split()
                np_array = np.array(np_array_str).astype(member_dict[member].dtype)

                # extract shape, if shape is stored as attrib, reshape the array
                if 'shape' in attrib.keys():
                    np_shape_str = attrib['shape']
                    np_shape = tuple(map(int, np_shape_str.split()))
                    np_size = 1
                    for x in np_shape:
                        np_size *= x
                    if np_size == np_array.size:
                        np_array = np_array.reshape(np_shape)
                    else:
                        print(
                            'Error: ParseXmlNode (%s), shape not match, just read as flat, data size (%d), shape size (%d)' % (
                                member, np_array.size, np_size))

                # save this array
                setattr(self, member, np_array)

            # the node is list
            elif type(member_dict[member]) is list or type(member_dict[member]) is tuple:
                # validate len
                list_elem = child.findall('elem')
                list_len = len(list_elem)
                if 'len' in attrib.keys():
                    list_len = int(attrib['len'])
                    if len(list_elem) != list_len:
                        print('Error: ParseXmlNode (%s), len not match, use xml len, data len (%d), xml len (%d)' % (
                            member, len(list_elem), list_len))

                # create list
                list_data = [''] * list_len

                for i in range(len(list_elem)):
                    elem = list_elem[i]
                    elem_str = ''
                    if elem.text is not None:
                        elem_str = elem.text

                        # get index of this element
                    index = i
                    if 'index' in elem.attrib.keys():
                        index = int(elem.attrib['index'])
                        if index < 0 or index >= len(list_elem):
                            index = i

                    if 'type' in elem.attrib.keys():
                        if elem.attrib['type'] == type(None).__name__:
                            list_data[index] = None
                        elif locate(elem.attrib['type']) is None:
                            list_data[index] = elem_str
                        else:
                            list_data[index] = locate(elem.attrib['type'])(elem_str)
                    else:
                        list_data[index] = elem_str

                if type(member_dict[member]) is list:
                    member_dict[member] = list_data
                else:
                    member_dict[member] = tuple(list_data)

            # the node is None
            elif type(member_dict[member]) is type(None):
                setattr(self, member, None)

            # the node is simple python data
            else:
                setattr(self, member, type(member_dict[member])(child.text))

    def read_xml_from_file(self, filename):
        try:
            parser = xml.etree.ElementTree.parse(filename)
            node_root = parser.getroot()
        except:
            return False

        self.parse_xml_node(node_root)
        return True

    def write_xml_to_file(self, filename, xml_node=None):
        if xml_node is None:
            xml_node = self.create_xml_node()
        if xml_node is None:
            return False

        with open(filename, "w") as f:
            xml_ugly_str = xml.etree.ElementTree.tostring(xml_node, 'utf-8')
            xml_parser = xml.dom.minidom.parseString(xml_ugly_str)
            xml_pretty_str = xml_parser.toprettyxml()
            f.write(xml_pretty_str)
            return True

        return False

    def print_info(self):
        print('========== %s ==========' % self.__class__.__name__)
        member_dict = vars(self)
        member_name = list(vars(self).keys())

        for member in member_name:
            # the data is numpy array
            if type(member_dict[member]) is np.ndarray:
                if member_dict[member].size != 0:
                    flatten = member_dict[member].reshape(member_dict[member].size)
                    print(member + '(%s) : [shape: %s, min: %f, max: %f, mean: %f, dtype: %s]' % (
                        type(member_dict[member]).__name__,
                        str(member_dict[member].shape),
                        flatten.min(),
                        flatten.max(),
                        flatten.mean(),
                        str(member_dict[member].dtype)
                    ))
                else:
                    print(member + '(%s) : [shape: %s, dtype: %s]' % (
                        type(member_dict[member]).__name__,
                        str(member_dict[member].shape),
                        str(member_dict[member].dtype)
                    ))

            # the data is list
            elif type(member_dict[member]) is list:
                if len(member_dict[member]) != 0:
                    print(member + '(%s) : [len: %d, first: %s(%s)]' % (
                        type(member_dict[member]).__name__,
                        len(member_dict[member]),
                        str(member_dict[member][0]),
                        type(member_dict[member][0]).__name__
                    ))
                else:
                    print(member + '(%s) : [len: %d]' % (
                        type(member_dict[member]).__name__,
                        len(member_dict[member])
                    ))

            # the data is basic python type
            else:
                print(member + '(%s) : %s' % (
                    type(member_dict[member]).__name__,
                    str(member_dict[member])
                ))

