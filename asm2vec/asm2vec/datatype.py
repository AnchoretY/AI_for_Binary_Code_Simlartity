import torch
import random
import warnings

class Token:
    def __init__(self, name, index):
        self.name = name    # token字符串
        self.index = index  # 索引
        self.count = 1      # 频率
    def __str__(self):
        return self.name

class Tokens:
    def __init__(self, name_to_index=None, tokens=None):
        self.name_to_index = name_to_index or {} # token字符串到index的映射字典
        self.tokens = tokens or []               # tokens对象（token组成的列表）
        self._weights = None                     # 负采样的频率 

    def __getitem__(self, key):
        if type(key) is str:
            if self.name_to_index.get(key) is None:
                warnings.warn("Unknown token in training dataset")
                return self.tokens[self.name_to_index[""]]
            return self.tokens[self.name_to_index[key]]
        elif type(key) is int:
            return self.tokens[key]
        else:
            try:
                return [self[k] for k in key]
            except:
                raise ValueError
    def load_state_dict(self, sd):
        self.name_to_index = sd['name_to_index']
        self.tokens = sd['tokens']
    def state_dict(self):
        return {'name_to_index': self.name_to_index, 'tokens': self.tokens}

    # tokens中包含的token个数
    def size(self):
        return len(self.tokens)

    # 
    def add(self, names):
        """
            输入为token字符串列表，如果已经在tokens中已经包含token，那么token的count+1，如果没有包含的token，那么新建token，索引为递增
        """
        self._weights = None
        if type(names) is not list:
            names = [names]
        for name in names:
            # name为新的
            if name not in self.name_to_index:
                token = Token(name, len(self.tokens))
                self.name_to_index[name] = token.index
                self.tokens.append(token)
            else:
                self.tokens[self.name_to_index[name]].count += 1
    
    def update(self, tokens_new):
        """
            输入为token对象列表，功能与add相同，将new_tokens中的新token加入当前tokens
        """
        for token in tokens_new:
            if token.name not in self.name_to_index:
                token.index = len(self.tokens)
                self.name_to_index[token.name] = token.index
                self.tokens.append(token)
            else:
                self.tokens[self.name_to_index[token.name]].count += token.count

    def weights(self):
        """
            根据每个token的频数计算各个token的采样权重
        """
        # if no cache, calculate
        if self._weights is None:
            total = sum([token.count for token in self.tokens])
            self._weights = torch.zeros(len(self.tokens))
            for token in self.tokens:
                self._weights[token.index] = (token.count / total) ** 0.75
        return self._weights

    def sample(self, batch_size, num=5):
        """
            随机负采样，默认一个batch中的每个样本生成5个随机采样到的负样本
        """
        return torch.multinomial(self.weights(), num * batch_size, replacement=True).view(batch_size, num)


class Function:
    def __init__(self, insts, blocks):
        self.insts = insts  # 函数包含的指令对象列表
        self.blocks = blocks
        
    @classmethod
    def load(cls, cfg_data_dict):
        '''
        gcc -S format compatiable
        '''
        label, labels, insts, blocks = None, {}, [], []
        block_map = {}

        for block_dict in cfg_data_dict:
            block_address = block_dict['addr']
            block = BasicBlock()
            block_map[block_address] = block

            for line in block_dict['text']:
                ins_string = line['text']
                inst = Instruction.load(ins_string)
                # 指令
                block.add(inst)
                insts.append(inst)

            blocks.append(block)
            # 将当前基本块连接在之前基本块之后
            if len(blocks) > 1:
                blocks[-2].successors.add(blocks[-1])

        # 基本块之间建立链接
        for block in blocks:
            inst = block.insts[-1]
            if inst.is_jmp() and block_map.get(inst.args[0]):
                block.successors.add(block_map[inst.args[0]])

        # replace label with CONST
        for inst in insts:
            for i, arg in enumerate(inst.args):
                if labels.get(arg):
                    inst.args[i] = 'CONST'
        return cls(insts, blocks)

    # 函数中包含的指令所包含的token string列表，多条指令出现重复token string重复出现token string，且包操作数不足两个填充造成的空token string
    def tokens(self):
        return [token for inst in self.insts for token in inst.tokens()]

    def random_walk(self, num=3):
        """
            在函数上进行随机游走产生的ins对象序列，随机游走的长度不固定，随机
                nums:在该函数上生成随机游走序列的条数，
        """
        return [self._random_walk() for _ in range(num)]

    def _random_walk(self):
        current, visited, seq = self.blocks[0], [], []
        while current not in visited:
            visited.append(current)
            seq += current.insts
            # no following block / hit return
            if len(current.successors) == 0 or current.insts[-1].op == 'ret':
                break
            current = random.choice(list(current.successors))
        return seq
        
class BasicBlock:
    def __init__(self):
        self.insts = []
        self.successors = set()
    def add(self, inst):
        self.insts.append(inst)

    # 判断当前基本块的最后一个指令是否为
    def end(self):
        inst = self.insts[-1]
        return inst.is_jmp() or inst.op == 'ret'

class Instruction:
    def __init__(self, op, args):
        self.op = op            # 操作码，字符串
        self.args = args        # 操作数，list，固定包含两个元素，没有操作数填充""

    def __str__(self):
        return f'{self.op} {", ".join([str(arg) for arg in self.args if str(arg)])}'
    @classmethod
    def load(cls, text):
        text = text.strip().strip('bnd').strip() # get rid of BND prefix
        op, _, args = text.strip().partition(' ')
        if args:
            args = [arg.strip() for arg in args.split(',')]
        else:
            args = []
        args = (args + ['', ''])[:2] # 保证两个参数
        return cls(op, args)

    # 获取指令的指令的tokens列表，三元组[op,opcode1,opcode2]
    def tokens(self):
        return [self.op] + self.args

    def is_jmp(self):
        return 'jmp' in self.op or self.op[0] == 'j'
    def is_call(self):
        return self.op == 'call'
