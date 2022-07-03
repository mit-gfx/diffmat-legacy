'''
Function graph definitions for SBS file parser.
'''
import abc
import operator
import numpy as np
import xml.etree.ElementTree as ET

from collections import deque

class SBSFunctionGraphParser:
    """The wrapper class of a SBS function graph parser for a dynamic node parameter.
    """
    def __init__(self, root, param_dict, global_name=''):
        """
        Args:
            root (ET.Element): XML element of the dynamic parameter
            param_dict (dict): dictionary of exposed parameters
            global_name (str, optional): global and unambiguous name of the dynamic parameter. Defaults to '' (not assigned).
        """
        # Initialize node dictionary
        self.nodes = {}

        # Root node
        self.root_node = None

        # Initialize parameter name
        self.global_name = global_name

        # Optimization flag
        # If true, the dynamic parameter is linked to at least one trainable exposed parameter
        # and requires gradient
        self.trainable = False

        # Initialize dictionaries for constants and functions
        self.init_dicts()

        # Parse graph content from ET node
        self.parse(root, param_dict)

        # Dataflow analysis
        self.analyze()

    def init_dicts(self):
        """Initialize the dictionaries for type lookup.
        """
        # Default value dictionary for constant nodes
        # Format: tag -> default value
        self.const_dict = {
            'integer1': 0,
            'integer2': [0, 0],
            'integer3': [0, 0, 0],
            'integer4': [0, 0, 0, 0],
            'int1': 0,
            'float1': 0.0,
            'float2': [0.0, 0.0],
            'float3': [0.0, 0.0, 0.0],
            'float4': [0.0, 0.0, 0.0, 0.0],
            'bool': False,
            'string': '',
        }

        # Helpers for function data
        def swizzle(x, d):
            d = np.array(d)
            if not isinstance(x, np.ndarray):
                x = np.array([x])
            ret = np.where(d < len(x), x[np.minimum(d, len(x) - 1)], 0.0)
            return ret if ret.size > 1 else ret.item()

        vector = lambda x, y: np.hstack((x, y))
        array_cast_float = lambda x: x.astype(np.float64)
        array_cast_int = lambda x: x.astype(np.int64)

        swizzle_data = lambda dim: (list(range(dim)) if dim > 1 else 0, swizzle, ['vector'], 'swizzle({}, {})')
        vector_data = (None, vector, ['componentsin', 'componentslast'], 'vector({}, {})')
        lerp_op_data = lambda func: (None, func, ['a', 'b', 'x'], 'th.lerp({}, {}, {})')
        ifelse_op_data = lambda func: (None, func, ['condition', 'ifpath', 'elsepath'], 'th.where({}, {}, {})')

        unary_op_data = lambda func, template: (None, func, ['a'], template)
        binary_op_data = lambda func, template: (None, func, ['a', 'b'], template)
        cast_op_data = lambda func, template: (None, func, ['value'], template)

        # Default function data dictionary for math operation nodes (add as needed)
        # Format: tag -> (default data, default func, input name sequence)
        #      or tag -> {dtype1: (...), dtype2: (...)}
        self.op_dict = {}

        # Swizzle op
        for i in (1, 2, 3, 4):
            self.op_dict[f'swizzle{i}'] = swizzle_data(i)
            self.op_dict[f'iswizzle{i}'] = swizzle_data(i)

        # Vector op
        for i in (2, 3, 4):
            self.op_dict[f'vector{i}'] = vector_data
            self.op_dict[f'ivector{i}'] = vector_data

        # Cast op
        self.op_dict['tofloat'] = cast_op_data(float, '{}.float()')
        self.op_dict['tointeger'] = cast_op_data(int, '{}.long()')
        for i in (2, 3, 4):
            self.op_dict[f'tofloat{i}'] = cast_op_data(array_cast_float, '{}.float()')
            self.op_dict[f'tointeger{i}'] = cast_op_data(array_cast_int, '{}.long()')

        # lerp
        lerp = lambda a,b,x: a + (b-a)*x
        ifelse = lambda condition, ifpath, elsepath: ifpath if condition else elsepath

        def rand(x):
            print('Warning: random function in node parameter.')
            return np.random.uniform(high=x)

        # Other ops
        self.op_dict.update({
            'toint1': unary_op_data(int, '{}.long()'),
            'sub': binary_op_data(operator.sub, '{} - {}'),
            'add': binary_op_data(operator.add, '{} + {}'),
            'mul': binary_op_data(operator.mul, '{} * {}'),
            'div': {16: binary_op_data(operator.floordiv, '{} // {}'),
                    256: binary_op_data(operator.truediv, '{} / {}')},
            'neg': unary_op_data(operator.neg, '-{}'),
            'max': binary_op_data(max, 'th.max({}, {})'),
            'min': binary_op_data(min, 'th.min({}, {})'),
            'abs': unary_op_data(abs, 'th.abs({})'),
            'eq':  binary_op_data(operator.eq, '{} == {}'),
            'lerp': lerp_op_data(lerp),
            'ifelse': ifelse_op_data(ifelse),
            'mod': binary_op_data(operator.mod, '{} % {}'),
            'lreq': binary_op_data(operator.le, '{} <= {}'),
            'gteq': binary_op_data(operator.ge, '{} >= {}'),
            'rand': unary_op_data(rand, 'th.rand([]) * {}')
        })

    def lookup_type(self, tag, dtype):
        """Obtain class name, default data, and function via node tag and data type
        """
        if 'const' in tag:
            node_class = SBSFunctionConstant
            node_args = self.const_dict[tag[tag.rfind('_') + 1:]],
        elif 'get' in tag:
            node_class = SBSFunctionGet
            node_args = None,
        else:
            node_class = SBSFunctionOp
            ret = self.op_dict[tag]
            node_args = ret[dtype] if isinstance(ret, dict) else ret
        return node_class, node_args

    def resolve_param_val(self, param_value_node):
        """Extract the default value of a parameter input according to its type ID.
        """
        param_value_ = param_value_node
        param_tag = param_value_.tag
        if param_tag in ['constantValueInt32', 'constantValueInt1']:
            param_val = int(param_value_.get('v'))
        elif param_tag in ['constantValueInt2', 'constantValueInt3', 'constantValueInt4']:
            param_val = [int(i) for i in param_value_.get('v').strip().split()]
        elif param_tag == 'constantValueFloat1':
            param_val = float(param_value_.get('v'))
        elif param_tag in ['constantValueFloat2', 'constantValueFloat3', 'constantValueFloat4']:
            param_val = [float(i) for i in param_value_.get('v').strip().split()]
        elif param_tag == 'constantValueBool':
            param_val = bool(int(param_value_.get('v')))
        elif param_tag == 'constantValueString':
            param_val = param_value_.get('v')
        else:
            raise TypeError('Unknown parameter type')
        return param_val

    def parse(self, root, param_dict):
        """Parse the function graph.

        Args:
            root (ET.Element): root XML element of the function graph
            param_dict (dict): dictionary of exposed parameters
        """
        # Placeholder creator
        def create_placeholder(name):
            return lambda: name

        # Locate nodes
        for node_ in root.iter('paramNode'):
            # Basic node information
            node_uid = int(node_.find('uid').get('v'))
            node_tag = node_.find('function').get('v')
            node_dtype = int(node_.find('type').get('v'))
            if node_dtype == None:
                node_dtype = node_.find('type/value').get('v')
            node_dtype = int(node_dtype)

            # Build function node (constant or operation)
            node_class, node_args = self.lookup_type(node_tag, node_dtype)
            node_obj = node_class(node_uid, node_dtype, node_tag, *node_args)

            # Resolve function data
            func_data = []
            for func_data_ in node_.findall('funcDatas/funcData'):
                func_data_name = func_data_.find('name').get('v')
                func_data_val_ = func_data_.find('constantValue')[0]
                func_data_val = self.resolve_param_val(func_data_val_)

                # Get global parameter value (and skip node variable for now)
                if 'get' in func_data_name:
                    if func_data_val in param_dict:
                        input_param = param_dict[func_data_val]
                        node_obj.name = func_data_val
                        node_obj.trainable = input_param.is_trainable()
                        func_data_val = type_cast(input_param.val, node_dtype)
                    else:
                        func_data_val = create_placeholder(func_data_val)

                func_data.append((func_data_name, func_data_val))

            # Update function data
            if func_data:
                node_obj.update_data(func_data)

            # Add to node dictionary
            self.nodes[node_uid] = node_obj

        # Obtain root node reference
        root_node_ = root.find('rootnode')
        if root_node_ is not None:
            root_node = int(root_node_.get('v'))
            self.root_node = self.nodes[root_node]
        else:
            raise RuntimeError('Function graph without root node')

        # Scan graph connectivity
        for node_ in root.iter('paramNode'):
            node_uid = int(node_.find('uid').get('v'))
            node_obj = self.nodes[node_uid]
            for conn_ in node_.findall('connections/connection'):
                conn_id = conn_.find('identifier').get('v')
                conn_ref = int(conn_.find('connRef').get('v'))
                node_obj.add_connection(conn_id, self.nodes[conn_ref])

    def analyze(self):
        """Run data flow analysis to find active nodes and determine node sequence.
        """
        # Calculate reachable nodes from the output
        queue = deque([self.root_node])
        active_uids = {self.root_node.uid}

        # Run backward BFS
        while queue:
            node = queue.popleft()
            for _, conn_ref in node.connections:
                if conn_ref is not None and conn_ref.uid not in active_uids:
                    queue.append(conn_ref)
                    active_uids.add(conn_ref.uid)

        # Save visited nodes
        active_nodes = [self.nodes[uid] for uid in active_uids]
        active_get_nodes = [node for node in active_nodes if isinstance(node, SBSFunctionGet)]
        active_const_nodes = [node for node in active_nodes if isinstance(node, SBSFunctionConstant)]

        # Check if this dynamic parameter is trainable
        self.trainable = any([node.trainable for node in active_get_nodes])

        # Count in-degrees
        indegrees = {node.uid: len([conn_ref.uid for _, conn_ref in node.connections \
                                    if conn_ref.uid in active_uids]) for node in active_nodes}

        # Topology sorting
        self.node_seq = []
        queue.extend(active_get_nodes)
        queue.extend(active_const_nodes)
        while queue:
            node = queue.popleft()
            self.node_seq.append(node)
            for target in active_nodes:
                for _, conn_ref in target.connections:
                    if conn_ref == node:
                        indegrees[target.uid] -= 1
                        if not indegrees[target.uid]:
                            queue.append(target)

        # Check validity
        assert self.root_node == self.node_seq[-1], 'Output node is not in the sequence.'

        # Assign variable names
        for i, node in enumerate(self.node_seq):
            node.var_name = f'x{i}'

    def mark_exposed_params(self, param_dict):
        """Mark used exposed parameters if the node parameter is trainable.
        """
        if self.trainable:
            for node in self.node_seq:
                if isinstance(node, SBSFunctionGet) and node.name in param_dict:
                    input_param = param_dict[node.name]
                    input_param.used = True

    def eval(self, node_param_dict):
        """Evaluate the result of the function graph via recursion.

        Args:
            node_param_dict (dict): dictionary of internal node parameters

        Returns:
            Any: output value of the function graph
        """
        # Fill in missing node parameters in 'get' nodes
        for _, node in self.nodes.items():
            if 'get' in node.tag and callable(node.data):
                node.data = node_param_dict[node.data()]

        # Evalulate function graph
        return type_cast(self.root_node.eval(), self.root_node.dtype)

    def get_func_str(self):
        """Write the function definition.
        """
        # Special case for getting a value only
        if isinstance(self.root_node, SBSFunctionGet) and not self.root_node.name.startswith('$'):
            return f"{self.global_name} = itemgetter_torch('{self.root_node.name}')"

        # Function header
        header = f'def {self.global_name}(params):'
        str_list = [header]

        # Function description
        for node in self.node_seq:
            str_list.append(f'    {node.get_op_str()}')
        str_list.append(f'    return {self.root_node.var_name}.squeeze()')
        str_list.append('')

        return '\n'.join(str_list)

class SBSFunctionNode(abc.ABC):
    """The wrapper class of a function graph node.
    """
    def __init__(self, uid, dtype, tag, data):
        """
        Args:
            uid (int): node UID
            dtype (int): data type associated to the node
            tag (str): node type
            data (Any): embedded parameters/operands
        """
        # Basic information
        self.uid = uid
        self.dtype = dtype
        self.tag = tag
        self.data = data
        self.var_name = ''

        # Input connections
        self.connections = []

    # Add an input connection
    def add_connection(self, name, node_ref):
        self.connections.append((name, node_ref))

    # Obtain an input connection indexed by name
    def get_connection(self, name):
        for conn_name, conn_ref in self.connections:
            if conn_name == name:
                return conn_ref
        raise RuntimeError(f"Connection '{name}' is not found in node {self.uid}")

    # Check if a connection exists, specified by the target
    def has_connection_target(self, node):
        return node in [conn_ref for _, conn_ref in self.connections]

    # Update function data
    def update_data(self, data):
        if isinstance(self.data, dict):
            for key, val in data:
                self.data[key] = val
        else:
            self.data = data[0][1]

    @abc.abstractmethod
    def eval(self):
        """Evaluate the result at the current node.
        """
        return None

    @abc.abstractmethod
    def get_op_str(self):
        """Write the operation in string format
        """
        return None

class SBSFunctionConstant(SBSFunctionNode):
    """A constant value from 'const' node or 'get' node.
    """
    def eval(self):
        return np.array(self.data) if isinstance(self.data, list) else self.data

    def get_op_str(self):
        return ' = '.join([self.var_name, to_str(self.data)])

class SBSFunctionGet(SBSFunctionNode):
    """A reference to an exposed parameter or a node parameter
    """
    def __init__(self, uid, dtype, tag, data, name='', trainable=False):
        """
        Args:
            uid (int): node UID
            dtype (int): node data type
            tag (str): node type
            data (Any): embedded parameters/operands
            name (str, optional): name of the target parameter. Defaults to '' (unassigned).
            trainable (bool, optional): switch indicating whether the target parameter is trainable. Defaults to False.
        """
        super().__init__(uid, dtype, tag, data)
        self.name = name
        self.trainable = trainable

    def eval(self):
        return np.array(self.data) if isinstance(self.data, list) else self.data

    def get_op_str(self):
        return ' = '.join([self.var_name,
                           f"th.as_tensor(params['{self.name}'])" if self.name else to_str(self.data)])

class SBSFunctionOp(SBSFunctionNode):
    """An operation on one or more inputs.
    """
    def __init__(self, uid, dtype, tag, data, func, input_seq, func_template):
        """
        Args:
            uid (int): node UID
            dtype (int): node data type
            tag (str): node type
            data (Any): embedded parameters/operands
            func (callable): operation function
            input_seq (list): list of input operands
            func_template (callable): template of function description in string
        """
        super().__init__(uid, dtype, tag, data)
        self.func = func
        self.input_seq = input_seq
        self.template = func_template

    def eval(self):
        args = [self.get_connection(name).eval() for name in self.input_seq]
        if self.data is not None:
            args.append(self.data)
        return self.func(*args)

    def get_op_str(self):
        args = [self.get_connection(name).var_name for name in self.input_seq]
        assert all(args), 'Undetected input variable.'
        if self.data is not None:
            args.append(to_str(self.data))
        return ' = '.join([self.var_name, self.template.format(*args)])

### Helper functions

def type_cast(val, dtype):
    """Type checking and casting.
    """
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if dtype == 4 and not isinstance(val, bool):
        val = bool(val)
    elif dtype == 16 and not isinstance(val, int):
        val = int(val)
    elif dtype in (32, 64, 128) and isinstance(val, list) and not isinstance(val[0], int):
        val = [int(i) for i in val]
    elif dtype == 256 and not isinstance(val, float):
        val = float(val)
    elif dtype in (512, 1024, 2048) and isinstance(val, list) and not isinstance(val[0], float):
        val = [float(i) for i in val]
    elif dtype == 16384 and not isinstance(val, str):
        val = str(val)
    return val

def to_str(val):
    """Convert a value to its PyTorch tensor string format
    """
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, (int, bool, float, list, tuple)):
        return f'th.tensor({val})'
    elif isinstance(val, str):
        return f"'{val}'"
    else:
        raise TypeError('Unrecognized value type')
