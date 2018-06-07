#encoding=utf-8
import tensorflow as tf
import numpy as np
from embedding_flag_cnntext_fenlei import data_helper,cnn_text_model
'''
进行反向传播，优化模型
'''
# 模型的超参数
tf.flags.DEFINE_integer("vector_size", 80, "每个单词向量的维度")
tf.flags.DEFINE_integer("sentence_length", 50, "句子的长度")
tf.flags.DEFINE_integer("num_filters", 128, "卷积核的个数")
tf.flags.DEFINE_integer("num_classes", 2, "类别种类数")
tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "L2正则化系数的比率")
filter_hs=[3,4,5]
# 训练参数
tf.flags.DEFINE_float("keep_prob", 0.5, "丢失率")
tf.flags.DEFINE_integer("batch_size", 30, "每个批次的大小")
tf.flags.DEFINE_integer("num_epochs", 10, "训练的轮数")
tf.flags.DEFINE_integer("num_steps", 1000, "学习率衰减的步数")
tf.flags.DEFINE_float("init_learning_rate", 0.01, "初始学习率")
FLAGS = tf.flags.FLAGS
#注意这里必须是tf.int32类型的，因为是词的索引为整型
x = tf.placeholder(tf.int32, [None, FLAGS.sentence_length], name='input')
y = tf.placeholder('float', [None, FLAGS.num_classes], name='output')
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
def backward_propagation():
    # 首先是embedding层获得词向量数据
    with tf.name_scope("embedding"):
        word_vecs, _ = data_helper.word_pro(FLAGS.vector_size)
        input_x = tf.nn.embedding_lookup(word_vecs, x)
        #注意此处只能用tf.expand_dims()不能用np.expand_dims()，因为此处还没feed进去值
        input_x = tf.expand_dims(input_x, -1)
    # 初始化模型
    cnn = cnn_text_model.cnn_model(FLAGS.sentence_length, FLAGS.vector_size, FLAGS.num_filters, FLAGS.num_classes,
                                   filter_hs, input_x, keep_prob)
    y_pred, l2_loss = cnn.positive_propagation()
    # 获得类别标签的one_hot编码
    label = data_helper.get_onehot_label()
    label = np.array(label)
    # 计算损失值
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_pred)
    losses = tf.reduce_mean(loss) + FLAGS.l2_reg_lambda * l2_loss
    tf.summary.scalar('loss_function', losses)

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('accuracy_function', accuracy)
    # train-modle========================================
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(FLAGS.init_learning_rate, global_step, FLAGS.num_steps,
                                               0.09)  # 学习率递减
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(losses, global_step=global_step)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        merged_summary_op = tf.summary.merge_all()  # 合并所有summary
        # 创建summary_writer，用于写文件
        summary_writer = tf.summary.FileWriter('log/text_cnn_summaries', sess.graph)
        #划分训练集和测试集，注意此处是单词的索引并不是单词对应的向量
        sen_index= data_helper.get_index_array(FLAGS.sentence_length,FLAGS.vector_size)
        X_train = sen_index[:-30]
        X_test = sen_index[-30:]
        y_train = label[:-30]
        y_test = label[-30:]
        # #批量获得数据
        num_inter = int(len(y_train) / FLAGS.batch_size)
        for ite in range(FLAGS.num_epochs):
            for i in range(num_inter):
                start = i * FLAGS.batch_size
                end = (i + 1) * FLAGS.batch_size
                feed_dict={x:X_train[start:end],y:y_train[start:end],keep_prob:FLAGS.keep_prob}
                # 生成summary
                summary_str = sess.run(merged_summary_op,feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, ite)  # 将summary 写入文件
                if i % 10 == 0:
                    train_accuracy = accuracy.eval(
                        feed_dict={x:X_train[start:end],y:y_train[start:end],keep_prob:1.0})
                    print("Step %d accuracy is %f" % (i, train_accuracy))
                sess.run(train_step, feed_dict=feed_dict)
        print("test accuracy %g" % accuracy.eval(feed_dict={x:X_test,y:y_test,keep_prob:1.0}))
if __name__ == '__main__':
    backward_propagation()