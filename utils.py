import subprocess
import re
from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

# 存放op操作符字符串的位置

good_ops_dir = 'data/good_op.csv'
bad_ops_dir = 'data/bad_op.csv'

# 存放各模型的路径

lstm_dir='model/lstm'

#-- 参数区

# 选定数据集的路径，将在所有包含utils库的程序中使用该路径

# good_dir = 'data/test/good'
# bad_dir = 'data/test/bad'

good_dir='data/good'
bad_dir='data/bad'

embedding_size = 100  # 隐层的维度
vec_dir = "bin/word2vec.model"  # word2vec存放位置
window = 5  # 上下文距离
iter_num = 30  # word2vec的迭代数
min_num = 1  # word2vec的最少出现次数
max_voc = 1000  # 最大字典数
time_step = 30  # 时序，即单句的最大seg长

lens_dir = 'bin/lens.pkl'  # 存放所有payload的长度统计
y_train_dir = 'bin/y_train.npy'
x_train_dir = 'bin/x_train.npy'

#-- 参数区结束


def load_php_opcode(phpfilename):
    """
    获取php opcode 信息；提取一个php文件为opcode操作符连接的句子
    """
    try:
        output = str(subprocess.check_output(
            ['php.exe', '-dvld.active=1', '-dvld.execute=0', phpfilename], stderr=subprocess.STDOUT))

        # with open('test.txt','wb') as fw:
        #     fw.write(output)
        # exit()

        tokens = re.findall(r'\s(\b[A-Z_]+\b)\s', output)  # opcode操作符提取正则
        t = " ".join(tokens)
        return t.replace('E O E ', '')  # 由于opcode正则会匹配每个func开头的非opcode字符，在这里去除
    except Exception as e:
        print('[Error] ', phpfilename, ' Error: ', e)
        return ""  # 未读取成功或没有任何操作符时


def shuffle_data(train_data, train_target):
    batch_size = len(train_target)
    index = [i for i in range(0, batch_size)]
    np.random.shuffle(index)
    batch_data = []
    batch_target = []
    for i in range(0, batch_size):
        batch_data.append(train_data[index[i]])
        batch_target.append(train_target[index[i]])
    return batch_data, batch_target

