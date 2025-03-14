# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, List

from nvflare.fuel.common.fqn import FQN


def _build_path(obj: Any, path: list, get_name_f, get_parent_f, **kwargs):
    name = get_name_f(obj, **kwargs)
    if name in path:
        return f"circular parent ref {name}"

    path.insert(0, name)
    parent = get_parent_f(obj, **kwargs)
    if not parent:
        return ""

    return _build_path(parent, path, get_name_f, get_parent_f, **kwargs)


def build_path(obj: Any, get_name_f, get_parent_f, **kwargs):
    """Build tree path for the specified object, which can be of any type.
    It is assumed that the object is already in an object tree. The tree path of an object is the list of objects
    that are traversed from the root to the object.

    Objects must have unique names.

    Args:
        obj: the object that the path will be built for
        get_name_f: the function that returns object name
        get_parent_f: the function that returns the object's parent object
        **kwargs: kwargs to be passed to the get_name_f and get_parent_f

    Returns: a tuple of (error, path).

    """
    if not callable(get_name_f):
        raise ValueError("get_name_f is not callable")

    if not callable(get_parent_f):
        raise ValueError("get_parent_f is not callable")

    path = []
    err = _build_path(obj, path, get_name_f, get_parent_f, **kwargs)
    return err, path


class Node:
    def __init__(self, obj: Any, parent):
        self.obj = obj
        self.parent = parent  # a Node
        self.children = []  # child nodes


class Forest:
    def __init__(self):
        self.roots = []  # 1 or more roots
        self.nodes = {}  # name => Node


def build_forest(objs: List[Any], get_name_f, get_fqn_f, **kwargs) -> Forest:
    forest = Forest()
    f2n = {}  # fqn => name
    n2f = {}  # name => fqn
    f2o = {}  # fqn => Obj
    for obj in objs:
        fqn = get_fqn_f(obj, **kwargs)
        n = get_name_f(obj, **kwargs)
        n2 = f2n.get(fqn)
        if n2:
            raise ValueError(f"two names ({n} and {n2}) have the same FQN {fqn}")
        fqn2 = n2f.get(n)
        if fqn2:
            raise ValueError(f"two FQNs ({fqn} and {fqn2}) have the same name {n}")
        f2n[fqn] = n
        f2o[fqn] = obj
        n2f[n] = fqn

    fqn_to_nodes = {}
    for fqn, obj in f2o.items():
        parent_fqn = FQN.get_parent(fqn)
        node = Node(obj, parent_fqn)
        fqn_to_nodes[fqn] = node
        name = f2n[fqn]
        forest.nodes[name] = node

    # resolve fqns to nodes
    for name, node in forest.nodes.items():
        assert isinstance(node, Node)
        if node.parent:
            # node.parent is the fqn of the parent
            parent_node = fqn_to_nodes.get(node.parent)
            if not parent_node:
                raise ValueError(f"missing node definition for FQN {node.parent}, which is the parent of {name}")
            node.parent = parent_node
            parent_node.children.append(node)
        else:
            # this node has no parent - it's a root
            forest.roots.append(name)

    return forest


def _dump_one(node: Node, get_name_f, **kwargs):
    name = get_name_f(node.obj, **kwargs)
    if not node.children:
        return name
    children = []
    for n in node.children:
        children.append(_dump_one(n, get_name_f, **kwargs))
    return {name: children}


def dump_forest(forest: Forest, get_name_f, **kwargs):
    trees = []
    for r in forest.roots:
        node = forest.nodes[r]
        trees.append(_dump_one(node, get_name_f, **kwargs))

    return {"roots": forest.roots, "trees": trees}
