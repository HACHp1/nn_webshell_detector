from utils import *
from lstm_tf2 import *
from word2vec import *

w2v_model = Word2Vec.load(vec_dir)
model=tf.keras.models.load_model('model/lstm.keras') # 载入模型参数

'''
向量化函数
传入：单个webshell的源代码
传出：向量化的结果
'''


def tovector(payload):
    with open('etc/temp', 'w') as fw:
        fw.write(payload)

    payload_seged = load_php_opcode('etc/temp').split(' ')
    # print(payload_seged)

    temp_x=[]
    for word in payload_seged:
        try:
            temp_x.append(w2v_model.wv.get_vector(word))
        except KeyError as e:
            temp_x.append(np.zeros((embedding_size)))  # 若不在字典中，则输入0占位

    temp_x=np.array(temp_x)
     # 字符串向量长度填充
    lenth = time_step

    if (temp_x.shape[0] < lenth):
        temp_x = np.pad(temp_x, ((0, lenth - temp_x.shape[0]),
                                    (0, 0)), 'constant', constant_values=0)
    else:
        temp_x = temp_x[0:lenth]

    return temp_x


def lstm_detect(x_test):
    pred_y=model.predict(x_test)
    return np.argmax(pred_y)


if __name__ == '__main__':
    a = '<?php eval($_GET[123]); ?>'
    a_vec=tovector(a)
    # print(a_vec.shape)
    # print(a_vec)
    print(lstm_detect(a_vec.reshape(1,time_step,embedding_size)))
