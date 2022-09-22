import json
import logging
from collections import defaultdict
from itertools import zip_longest

from nltk.tokenize import RegexpTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Node():
    def __init__(self) -> None:
        self.type = None
        self.label = None
        self.pos = None
        self.children = []
        self.father = None
        self.idx = None

    def __str__(self) -> str:
        return f'{self.idx} {self.type} {self.pos} {self.father.idx if self.father else -1} {self.label}'

    def __repr__(self) -> str:
        return self.__str__()

    def unique_id(self) -> str:
        return '{} {}'.format(self.type, self.pos)

    def print_tree(self, depth):
        s = '    ' * depth + str(self) + '\n'
        for child in self.children:
            s += child.print_tree(depth + 1)
        return s


class SubWordTokenizer():
    """
    用于对标志符(下划线,字母,数字)进行分词得到一系列子词,一些分词实例:
    J2SE_CODE: [j, 2, se, code]
    RegexpTokenizer: [regexp, tokenizer]
    AST2NodePath: [ast, 2, node, path]
    现在支持保留特殊字符，如 a.B_C: [a . b c]
    """

    def __init__(self):
        self.tokenizer = RegexpTokenizer(pattern='[A-Za-z]+|\d+|[^\s\w]')
        self.alpha_tkn = RegexpTokenizer(pattern='[A-Z][a-z]*|[a-z]+')

    def tokenize(self, text):
        tokens = list()
        _tokens = self.tokenizer.tokenize(text)
        for i in _tokens:
            if i.isdigit() or i.islower():
                tokens.append(i)
            elif i.isupper() or len(i) == 1:
                tokens.append(i.lower())
            else:
                [tokens.append(j.lower()) for j in self.alpha_tkn.tokenize(i)]
        return tokens


def gen_ast(ast_str):
    try:
        ast = json.loads(ast_str)
    except:
        return []
    nodes = process_ast(ast['root'])
    cu = nodes[0]
    cu.idx = 0
    nodes_new = [cu]
    cur_node = cu
    idx = 1
    for node in nodes[1:]:
        if node == '^':
            cur_node = cur_node.father
        else:
            node.idx = idx
            node.father = cur_node
            cur_node.children.append(node)
            cur_node = node
            nodes_new.append(node)
            idx += 1
    return nodes_new


def process_ast(ast):
    nodes = []
    # if ast['type'] == 'Javadoc':
    #     return nodes
    node = Node()
    node.type = ast['type']
    node.pos = ast['pos']
    if 'label' in ast:
        node.label = ast['label']
    elif node.type == 'NullLiteral':
        node.label = 'null'
    elif node.type == 'ThisExpression':
        node.label = 'this'
    nodes.append(node)
    for child in ast['children']:
        ns = process_ast(child)
        if ns:
            nodes += ns
            nodes.append('^')
    return nodes


def get_node_idx_map(nodes):
    node_idx = {}
    for node in nodes:
        node_idx[node.unique_id()] = node.idx
    return node_idx


def process_act(act_str, node_idx1, node_idx2):
    """
    node_idx1是old tree
    node_idx2是new tree
    return: (matches, inserts, deletes, updates, moves)
    """
    act = json.loads(act_str)
    matches = dict()
    for m in act['matches']:
        if ':' not in m['src']:
            matches[node_idx1.get(m['src'], -1)] = node_idx2.get(m['dest'], -1)
        else:
            src = f"{m['src'].split(':', 1)[0]} {m['src'].rsplit(' ', 1)[-1]}"
            dest = f"{m['dest'].split(':', 1)[0]} {m['dest'].rsplit(' ', 1)[-1]}"
            matches[node_idx1.get(src, -1)] = node_idx2.get(dest, -1)
    inserts = set()
    deletes = set()
    updates = dict()
    moves = dict()
    for a in act['actions']:
        if a['action'].startswith('insert'):
            if ':' not in a['tree']:
                node = a['tree']
            else:
                node = f"{a['tree'].split(':', 1)[0]} {a['tree'].rsplit(' ', 1)[-1]}"
            inserts.add(node_idx2.get(node, -1))
        elif a['action'].startswith('delete'):
            if ':' not in a['tree']:
                node = a['tree']
            else:
                node = f"{a['tree'].split(':', 1)[0]} {a['tree'].rsplit(' ', 1)[-1]}"
            deletes.add(node_idx1.get(node, -1))
        elif a['action'].startswith('update'):
            if ':' not in a['tree']:
                node = a['tree']
                dest = a['dest']
            else:
                node = f"{a['tree'].split(':', 1)[0]} {a['tree'].rsplit(' ', 1)[-1]}"
                dest = f"{a['dest'].split(':', 1)[0]} {a['dest'].rsplit(' ', 1)[-1]}"
            updates[node_idx1.get(node, -1)] = node_idx2.get(dest, -1)
        elif a['action'].startswith('move'):
            if ':' not in a['tree']:
                node = a['tree']
                dest = a['dest']
            else:
                node = f"{a['tree'].split(':', 1)[0]} {a['tree'].rsplit(' ', 1)[-1]}"
                dest = f"{a['dest'].split(':', 1)[0]} {a['dest'].rsplit(' ', 1)[-1]}"
            moves[node_idx1.get(node, -1)] = node_idx2.get(dest, -1)
    return matches, inserts, deletes, updates, moves


def get_tokens_diffs(ast_nodes1, ast_nodes2, matches, inserts, deletes, updates, moves, code1, code2):
    """
    ast_nodes1是list of nodes of old tree
    ast_nodes2是list of nodes of new tree
    其他是process_act的返回结果
    return tokens1, tokens_idx1: token对应node的idx组成的list, tokens2, tokens_idx2, idxs1: 变更, idxs2, actions
    """
    nodes_idx1 = set(list(deletes) + list(updates.keys()) + list(moves.keys()))
    nodes_idx2 = set(list(inserts) + list(updates.values()) + list(moves.values()))
    context1 = get_context(ast_nodes1, nodes_idx1)
    context2 = get_context(ast_nodes2, nodes_idx2)
    tokens1 = []
    tokens_idx1 = []
    tokens_delete = []
    travel_collect(ast_nodes1[5], code1, context1, deletes, tokens1, tokens_idx1, tokens_delete, False, False)
    tokens2 = []
    tokens_idx2 = []
    tokens_insert = []
    travel_collect(ast_nodes2[5], code2, context2, inserts, tokens2, tokens_idx2, tokens_insert, False, False)
    update_idxs, move_idxs = get_updates_moves(ast_nodes1, ast_nodes2, tokens_idx1, tokens_idx2, updates, moves)
    idxs1, idxs2, actions = get_token_diff(tokens_insert, tokens_delete, update_idxs, move_idxs)
    return tokens1, tokens_idx1, tokens2, tokens_idx2, idxs1, idxs2, actions


def get_token_diff(inserts, deletes, updates, moves):
    """
    将前面各种方法处理得到的结果进行统合,输出idxs1, idxs2, actions分别对应tokens1中的idxs,tokens2中的idxs和对应操作
    inserts deletes是长度和tokens相同的list,True和False代表token是否添加和删除
    updates moves是二元组的list,元素是idx
    """
    actions = ['UPDATE', "MOVE", "INSERT", "DELETE"]
    triples = []
    # 处理顺序应该是update move insert delete
    idx1_set = set()
    idx2_set = set()
    for i, j in updates:
        triples.append((i, j, actions[0]))
        idx1_set.add(i)
        idx2_set.add(j)
    for i, j in moves:
        triples.append((i, j, actions[1]))
        idx1_set.add(i)
        idx2_set.add(j)
    for i, v in enumerate(inserts):
        if v and i not in idx2_set:
            idx2_set.add(i)
            triples.append((-1, i, actions[2]))
    for i, v in enumerate(deletes):
        if v and i not in idx1_set:
            idx1_set.add(i)
            triples.append((i, -1, actions[3]))
    triples.sort(key=lambda x: x[1] if x[1] > 0 else x[0] - 10000)
    return zip(*triples) if triples else ((), (), ())


def get_updates_moves(ast_nodes1, ast_nodes2, tokens_idx1, tokens_idx2, updates, moves):
    """
    给定两棵树,代码tokens对应的节点idx,以及updates和moves的列表
    返回在tokens上的updates和moves的操作
    """
    u_nodes1 = set(updates.keys())
    u_node_tokens1 = defaultdict(list)
    m_nodes1 = set(moves.keys())
    m_node_tokens1 = defaultdict(list)
    for i, v in enumerate(tokens_idx1):
        if v:
            node = ast_nodes1[v]
            while(node and node.idx > 5):
                if node.idx in u_nodes1:
                    u_node_tokens1[node.idx].append(i)
                    break
                elif node.idx in m_nodes1:
                    # 特殊处理move Block，一般对Block的move没有意义
                    if node.type == 'Block':
                        break
                    m_node_tokens1[node.idx].append(i)
                    break
                node = node.father
    u_nodes2 = set(updates.values())
    u_node_tokens2 = defaultdict(list)
    m_nodes2 = set(moves.values())
    m_node_tokens2 = defaultdict(list)
    for i, v in enumerate(tokens_idx2):
        if v:
            node = ast_nodes2[v]
            while(node and node.idx > 5):
                if node.idx in u_nodes2:
                    u_node_tokens2[node.idx].append(i)
                    break
                elif node.idx in m_nodes2:
                    # 特殊处理move Block，一般对Block的move没有意义
                    if node.type == 'Block':
                        break
                    m_node_tokens2[node.idx].append(i)
                    break
                node = node.father
    update_tokens = []
    move_tokens = []
    for n1, n2 in updates.items():
        tokens1 = u_node_tokens1[n1]
        tokens2 = u_node_tokens2[n2]
        # assert len(tokens1) == len(tokens2)
        for i, j in zip_longest(tokens1, tokens2, fillvalue=-1):
            update_tokens.append((i, j))
    for n1, n2 in moves.items():
        tokens1 = m_node_tokens1[n1]
        tokens2 = m_node_tokens2[n2]
        # assert len(tokens1) == len(tokens2)
        for i, j in zip_longest(tokens1, tokens2, fillvalue=-1):
            move_tokens.append((i, j))
    return update_tokens, move_tokens


def get_context(ast_nodes, modified_nodes_idx):
    """
    获取上下文,首先计算变更影响的变量,然后找到这些变量的def和use
    同时还需要找到控制依赖，找到所有控制语句相关，也作为上下文
    """
    context_node = set(modified_nodes_idx)
    # 找到变更影响的变量
    variables = set()  # 使用set保存，去重
    for i in modified_nodes_idx:
        if i == -1:
            continue
        node = ast_nodes[i]
        # 需要考虑的有"="号相关语句和方法调用
        # 影响只能从等号右边传到左边，从方法调用参数传到receiver
        # 首先肯定是遍历node及其子节点，找对直接变更的variables
        get_variables_in_node(node, variables, context_node)
        # 然后考虑等号和方法调用的传递
        get_variables_affected(node, variables)
    # 找到def和use语句作为上下文
    get_def_use_stmts(ast_nodes, variables, context_node)
    # 找到控制语句作为上下文
    get_control_stmts(ast_nodes, modified_nodes_idx, context_node)
    return context_node


def travel_collect(node, code, context_nodes, insert_delete_nodes, tokens, tokens_idx, tokens_i_d, c_flag, i_d_flag):
    """
    从根节点遍历一个AST树,收集context_nodes包含的tokens
    tokens_idx表示token在AST上的位置,被忽略的token对应idx为None
    本方法还会标注token的insert或delete属性,保存到tokens_i_d中
    """
    # 默认不处理Block
    if c_flag and node.type == 'Block':
        tokens.append('{')
        tokens_idx.append(None)
        tokens_i_d.append(False)
        for c in node.children:
            travel_collect(c, code, context_nodes, insert_delete_nodes, tokens, tokens_idx, tokens_i_d, False, i_d_flag)
        tokens.append('}')
        tokens_idx.append(None)
        tokens_i_d.append(False)
        return
    # TODO: 特殊处理SwitchStatement，（不常见，先不处理）
    # 判断节点是否属于context
    if node.idx in context_nodes:
        c_flag = True
    # 判断节点是否属于insert delete
    if node.idx in insert_delete_nodes:
        i_d_flag = True
    # 处理叶子结点
    if not node.children:
        if c_flag:
            if node.label:
                label = node.label
            else:
                s, e = parse_pos(node.pos)
                label = get_pos_of_code(code, s, e).strip()
            if label:
                tokens.append(label)
                tokens_idx.append(node.idx)
                tokens_i_d.append(i_d_flag)
        return
    # 正常处理其他类型的节点，省略掉的token一起加入
    cur, end = parse_pos(node.pos)
    for c in node.children:
        if c_flag:
            cs, ce = parse_pos(c.pos)
            label = get_pos_of_code(code, cur, cs).strip()
            if label:
                tokens.append(label)
                tokens_idx.append(None)
                tokens_i_d.append(False)
            cur = ce
        travel_collect(c, code, context_nodes, insert_delete_nodes, tokens, tokens_idx, tokens_i_d, c_flag, i_d_flag)
    if c_flag:
        label = get_pos_of_code(code, cur, end).strip()
        if label:
            tokens.append(label)
            tokens_idx.append(None)
            tokens_i_d.append(False)


def get_pos_of_code(code, s, e):
    return code[s - 19: e - 19]


def parse_pos(pos_str):
    s, e = pos_str[1:-1].split(',', 1)
    return int(s), int(e)


def get_variables_in_node(node, vars, context):
    """
    variables一定是SimpleName,反之不一定,需要去掉类名和方法名
    本方法还负责将变更node包含的语句加入到上下文中(效率)
    """
    if node.type == "SimpleName" and node.father and not node.father.type.endswith(("Type", 'MethodInvocation')):
        vars.add(node.label)
    if node.type.endswith(('Statement', 'CatchClause', 'SwitchCase')) and node.idx > 5:
        context.add(node.idx)
    for c in node.children:
        get_variables_in_node(c, vars, context)


def get_variables_affected(node, vars):
    par = node.father
    while(par):
        if par.type in ['Assignment', 'VariableDeclarationFragment', 'METHOD_INVOCATION_RECEIVER']:
            # 获取第一个SimpleName
            for c in par.children:
                if c.type == 'SimpleName':
                    vars.add(c.label)
                    break
        par = par.father


def get_def_use_stmts(ast_nodes, vars, stmts):
    """
    直接遍历ast_nodes,一旦遇到SimpleName的label在vars中,进行处理
    """
    # 因为某些原因，ast_nodes[5]一定是MethodDeclaration
    # MethodDeclaration必须添加到上下文stmts中
    assert ast_nodes[5].type == 'MethodDeclaration'
    stmts.add(5)
    for node in ast_nodes[6:]:
        if node.type == 'SimpleName' and node.label and node.label in vars:
            par = node.father
            # 找到node的最小的和Statement一个级别的node，将idx加入stmts
            while(par):
                if par.type.endswith(('Statement', 'CatchClause', 'SwitchCase')) and par.idx > 5:
                    stmts.add(par.idx)
                    break
                par = par.father


def get_control_stmts(ast_nodes, modified_nodes_idx, stmts):
    """
    遍历modified_nodes,找到相关的控制语句,if while switch do for [try]
    实际上就是一直搜索父节点,只要是Statement级别就加入
    """
    for i in modified_nodes_idx:
        if i == -1:
            continue
        node = ast_nodes[i]
        while(node):
            if node.idx <= 5:   # 跳出条件
                break
            if node.type.endswith(('Statement', 'CatchClause', 'SwitchCase')):
                stmts.add(node.idx)
            node = node.father  # 循环链


def process_tree_diff(obj):
    """
    obj是一个tree_diff的实例,是已经json.loads的一条数据
    return: tokens1, sub_tokens1, ast_atts1, tokens2, sub_tokens2, ast_atts2, idxs1, idxs2, actions
    """
    old_nodes = gen_ast(obj['tre1'])
    new_nodes = gen_ast(obj['tre2'])
    # 处理数据时，class层面的东西是补充的，真正的代码是方法定义，应该从nodes[5]开始
    if len(old_nodes) <= 5 or len(new_nodes) <= 5:
        return [], [], [], [], [], [], [], [], []
    code1 = obj['cod1']
    code2 = obj['cod2']
    node_idx_map1 = get_node_idx_map(old_nodes)
    node_idx_map2 = get_node_idx_map(new_nodes)
    act = process_act(obj['act'], node_idx_map1, node_idx_map2)
    res = get_tokens_diffs(old_nodes, new_nodes, act[0], act[1], act[2], act[3], act[4], code1, code2)
    lits = dict()
    type_num = defaultdict(int)
    sub_tokens1, ast_atts1 = code_rep(res[0], res[1], old_nodes, lits, type_num)
    sub_tokens2, ast_atts2 = code_rep(res[2], res[3], new_nodes, lits, type_num)
    return res[0], sub_tokens1, ast_atts1, res[2], sub_tokens2, ast_atts2, res[4], res[5], res[6]


def code_rep(tokens, node_idxs, nodes, lits, type_num):
    """
    将一个版本的代码处理成方便输入到神经网络模型的样子
    """
    tokenizer = SubWordTokenizer()
    # lits = dict()
    # type_num = defaultdict(int)
    sub_tokens = []
    ast_atts = []
    assert len(tokens) == len(node_idxs)
    for t, i in zip(tokens, node_idxs):
        if i:
            node = nodes[i]
            # t的类型为Literal
            if node.type.endswith('Literal'):
                if t in lits:
                    sub_tokens.append([lits[t], ])
                else:
                    type = node.type[:3]
                    num = type_num[type]
                    type_num[type] += 1
                    tn = f'{type}{num}'
                    lits[t] = tn
                    sub_tokens.append([tn, ])
            else:
                sub_tokens.append(tokenizer.tokenize(t))
            attr = []
            while node and node.idx >= 5:
                if node.type == 'MethodDeclaration':
                    attr.append('MD')
                    node = node.father
                    continue
                if node.type.endswith(('Statement', 'Block', 'CatchClause', 'SwitchCase')):
                    node_idx = node.father.children.index(node)
                    if node.type == 'Block':
                        attr.append(f'Bl{node_idx}')
                    elif node.type == 'CatchClause':
                        attr.append(f'Cat{node_idx}')
                    elif node.type == 'SwitchCase':
                        attr.append(f'Cas{node_idx}')
                    else:
                        attr.append(f'{node.type[:-9]}{node_idx}')
                node = node.father
            ast_atts.append(attr[::-1])
        else:
            sub_tokens.append(tokenizer.tokenize(t))
            ast_atts.append([])
    return sub_tokens, ast_atts


def print_token_diff(tokens1, tokens2, idx1, idx2, actions):
    for i, j, a in zip(idx1, idx2, actions):
        print(f'{tokens1[i] if i > 0 else None} {tokens2[j] if j > 0 else None} {a}')
