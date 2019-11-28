import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from utils import *
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from numpy import argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

n_inputs = embedding_size  # 输入维度，等于embedding大小
n_steps = time_step

n_classes = 2

input_keep_prob = 0.6  # dropout层
CONTINUE_TRAIN = True

# 超参数
lr = 0.01
training_iters = 100  # 迭代次数（不是epoch数） epoch=training_iters/data_len
BATCH_SIZE = 128  # 训练时批的大小

TEST_SIZE = 0.2
random_state = 0


'''
attention is all you need 论文中的 position embedding层
继承keras layer
h2 = PositionEmbedding(time_step, embedding_size)(h1)
'''


class PositionEmbedding(layers.Layer):

    def __init__(self, position, d_model, name="PositionEmbedding", **kwargs):
        super().__init__(name=name, **kwargs)

        # 储存用于恢复模型的init参数，用在get_config函数中
        #####
        self.position = position
        self.d_model = d_model
        #####

        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) /
                            tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'position': self.position,
            'd_model': self.d_model
        })
        return config


'''
keras层
为input添加一个可训练权重 => input · weight
'''


class Add_weight(layers.Layer):

    def __init__(self, name="Add_weight", **kwargs):  # 申请、储存本层需要用到的属性、对象等
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):  # 需要根据input_shape改变配置时要重写的函数
        # 添加的可训练的权重

        self.weight_to_mut = tf.Variable(tf.constant(
            0.1, shape=[input_shape[2], input_shape[2]]), trainable=True)
        super().build(input_shape)  # 一定要在最后调用它

    def call(self, x):  # 调用该层时进行的运算
        return K.dot(x, self.weight_to_mut)  # 点乘

    def get_config(self):  # 返回初始化变量，用于模型读取时使用
        config = super().get_config().copy()
        return config

    def compute_output_shape(self, input_shape):  # 返回的矩阵大小
        return (input_shape[2], input_shape[2])


def build_model():
    inputs = tf.keras.Input(shape=(n_steps, n_inputs,), batch_size=BATCH_SIZE)

    # weights

    input_wq = Add_weight(name='add_1')(inputs)
    input_wv = Add_weight(name='add_2')(inputs)
    input_wk = Add_weight(name='add_3')(inputs)

    h1 = layers.Attention()([
        input_wq,
        input_wv,
        input_wk
    ])  # self-attention [query,value,key]

    h2 = PositionEmbedding(time_step, embedding_size,
                           name="PositionEmbedding")(h1)  # 自定义层设置name

    h3 = layers.Flatten()(h2)  # 展开后使用全连接

    h4 = layers.Dense(n_classes, input_shape=(time_step, embedding_size))(h3)

    outputs = layers.Activation('softmax')(h4)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':

    if CONTINUE_TRAIN:
        # model = tf.keras.models.load_model('model/attention.keras', custom_objects={
        #     'PositionEmbedding': PositionEmbedding,
        #     'add_1':Add_weight,
        #     'add_2':Add_weight,
        #     'add_3':Add_weight,
        # })
        model = build_model()
        model.load_weights('model/attention_weights.keras')
    else:
        model = build_model()

    model.summary()

    values = np.array([0, 1])
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(integer_encoded)

    vx_train = np.load(x_train_dir)
    vy_train = np.load(y_train_dir)

    vy_train = onehot_encoder.transform(
        vy_train.reshape(-1, 1))  # 对y进行onehot编码

    x_train, x_test, y_train, y_test = train_test_split(
        vx_train, vy_train, test_size=TEST_SIZE, random_state=random_state)

    model.fit(x_train, y_train, batch_size=BATCH_SIZE,
              epochs=training_iters//x_train.shape[0], validation_split=0.1)

    # model.save('model/attention.keras')
    model.save_weights('model/attention_weights.keras', overwrite=True)

    y_pred = model.predict(x_test)
    y_pred_real = []
    for i in range(y_pred.shape[0]):
        y_pred_real.append(
            label_encoder.inverse_transform([argmax(y_pred[i])]))

    y_test_real = []
    for i in range(y_test.shape[0]):
        y_test_real.append(
            label_encoder.inverse_transform([argmax(y_test[i])]))  # 将y_test进行onehot还原

    print('test acc:', accuracy_score(y_test_real, y_pred_real))
    print('test recall:', recall_score(
        y_test_real, y_pred_real, average='macro'))
