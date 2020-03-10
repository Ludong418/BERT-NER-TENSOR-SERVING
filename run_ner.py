#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: dong.lu

@contact: ludong@cetccity.com

@software: PyCharm

@file: pre_process.py

@time: 2019/03/19 10:30

@desc: 运行ner模型

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import codecs
import pickle

import yaml
import tensorflow as tf

from bert import modeling
flags = tf.flags
FLAGS = flags.FLAGS

project_path = os.path.dirname(os.path.abspath(__file__))
from bert import optimization
from bert import tokenization

__version__ = '0.0.1'



flags.DEFINE_string("data_dir", project_path + '{}data'.format(os.sep), "The input datadir.")
flags.DEFINE_string("bert_config_file",
                    project_path + '{}chinese_L-12_H-768_A-12{}bert_config.json'.format(os.sep, os.sep),
                    "The config json file corresponding to the pre-trained BERT model.")
flags.DEFINE_string("task_name", 'ner', "The name of the task to train.")
flags.DEFINE_string("output_dir",
                    project_path + '{}checkpoint'.format(os.sep),
                    "The output directory where the model checkpoints will be written.")
flags.DEFINE_string('export_dir_base',
                    project_path + '{}model'.format(os.sep),
                    'The model type of PB file, when used SavedModel for saving a model')
flags.DEFINE_string("init_checkpoint",
                    project_path + os.sep + os.path.join('chinese_L-12_H-768_A-12', 'bert_model.ckpt'),
                    "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text.")
flags.DEFINE_integer("max_seq_length", 64, "The maximum total input sequence length after WordPiece tokenization.")
flags.DEFINE_bool("do_train", True, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run the model in predict mode on the test set.")
flags.DEFINE_bool("do_inference", False, "Whether to run the model in inference mode on online inference.")
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
flags.DEFINE_float("learning_rate", 5e-5/5, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 50.0, "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1,
                   "Proportion of training to perform linear learning rate warmup for. "
                   "E.g., 0.1 = 10% of training.")
flags.DEFINE_integer("save_checkpoints_steps", 500, "How often to save the model checkpoint.")
flags.DEFINE_integer("save_summary_steps", 500, 'save_checkpoints_steps')
flags.DEFINE_string("vocab_file",
                    project_path + '{}chinese_L-12_H-768_A-12{}vocab.txt'.format(os.sep, os.sep),
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool("use_one_hot_embeddings", True, 'use_one_hot_embeddings')


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    # TODO: 加如更多文件类型输入的处理方式
    @classmethod
    def read_data(cls, input_file):
        """Reads a BIO data."""
        with tf.gfile.Open(input_file, "r") as f:
            lines = []
            for line in f:
                line = line.strip()
                if line.startswith('-DOCSTART-'):
                    continue
                else:
                    word_labels = line.split('-seq-')
                    assert len(word_labels) == 2

                    words = word_labels[0]
                    labels = word_labels[1]
                    lines.append([words, labels])

        return lines


class NerProcessor(DataProcessor):
    # TODO： 加入输入文件检验
    def get_train_examples(self, data_dir=FLAGS.data_dir):
        return self._create_example(
            self.read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir=FLAGS.data_dir):
        return self._create_example(
            self.read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir=FLAGS.data_dir):
        return self._create_example(
            self.read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        yaml_path = project_path + '{}information.yaml'.format(os.sep)
        f = open(yaml_path)
        info = yaml.load(f, Loader=yaml.FullLoader)
        return info['label']

    @staticmethod
    def _create_example(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


def write_tokens(tokens, output_dir, mode):
    """
    在预测的时候，将token结果写入到文件中
    :arg
    tokens: token的列表，例如['a', 'b', '[SEP]']
    """
    if mode == "test":
        path = os.path.join(output_dir, "token_" + mode + ".txt")
        wf = codecs.open(path, 'a', encoding='utf-8')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode):
    """
    把一个example转换为feature，这一个过程会进行token的处理
    :arg
    ex_index: 是第几个example
    example: 单个InputExample的实例对像
    label_list: label的列表
    max_seq_length: 输入序列的最大的长度
    tokenizer: FullTokenizer的实例化对象
    output_dir: 模型存储路径
    mode: train/eval/test 模式
    """
    # label_list 映射到 int, 例如['']
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 检验label2id二进制文件是否存在
    if not tf.gfile.Exists(os.path.join(output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)
    else:
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'rb') as r:
            label_map = pickle.load(r)

    textlist = example.text.split(' ')
    labellist = example.label.split(' ')

    assert len(textlist) == len(labellist)

    tokens = []
    labels = []

    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        # 通过检验token的长度决定是否使用label_map中的标签
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")

    # 截断序列
    if len(tokens) >= max_seq_length - 1:
        # -2 的原因是因为序列需要加一个句首和句尾标志
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]

    # 把[CLS]，[SEP]加入到句首和句尾
    ntokens = []
    segment_ids = []
    label_ids = []
    # 句子开始设置CLS 标志
    ntokens.append("[CLS]")
    # 因为只有一句输入
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])

    ntokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])

    # 把token转化为id的形式
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    # 要注意的是input_mask是由1、0组成的，1表示真实的token，
    # 0表示填充的token，在这里并不是预训练的阶段，所以只能
    # 使用1来表示
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    # 打印前五个example
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    # 以InputFeatures对象存储example
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids)

    write_tokens(ntokens, output_dir, mode)

    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, output_dir, mode=None):
    """
    把examples转为features，并存到Tf_Record文件中
    :arg
    examples: InputExample的实例对像列表
    label_list: label的列表
    max_seq_length: 输入序列的最大的长度
    tokenizer: FullTokenizer的实例化对象
    output_file: Tf_Record文件
    output_dir: 模型存储路径
    mode: train/eval/test 模式
    """
    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        # 把每一个example表示成一个feature实例
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """
    Estimator的input_fn函数，用于从TFRecord中加载数据，然后进行类型上的变换，最后生成Dataset的batch数据
    input_file: 训练数据，TFRecord格式文件
    seq_length: int，每个输入的最大序列长度
    is_training: bool，是否是在训练
    drop_remainder: bool，是否删除最后一个batch
    """
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64)
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_calls=8,
            drop_remainder=drop_remainder
        ))
        d.prefetch(buffer_size=4)
        return d
    return input_fn


def serving_input_receiver_fn():
    """
    用于在serving时，接收数据
    :return:
    """
    feature_spec = {
        "input_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
    }
    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                           shape=[None],
                                           name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def create_model(bert_config, is_training, input_ids,
                 input_mask, segment_ids, labels, num_labels, use_one_hot_embeddings):
    """
    创建bert模型以及自定义创建输出层
    bert_config: bert超参数文件，定义在~/chinese_L-12_H-768_A-12/bert_config.json中
    is_training: 是否可训练
    input_ids: 输入的token的id， [batch_size, seq_length]
    input_mask: mask的编号， [batch_size, seq_length]  -> [1,1,1,1,1,...]
    segment_ids: 句类型的编码， [batch_size, seq_length] -> [0,0,0,0,...]
    labels: label的id，[batch_size, seq_length]
    num_labels: 标签的数量
    use_one_hot_embeddings: 是否使用one_hot编码
    """
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    # [batch_size, seq_length, hidden_size]
    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    # 输出层权重
    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    # 输出层偏质
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        loss = None
        per_example_loss = None
        # 展平 [batch_size*seq_length, hidden_size]
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        # 线性变换， [batch_size*seq_length, num_labels]
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        # [batch_size, seq_length, num_labels]
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])

        if FLAGS.do_train or FLAGS.do_eval or FLAGS.do_predict:
            # [batch_size, seq_length, num_labels]
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            # [batch_size, seq_length, num_labels]
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            # [batch_size, seq_length]
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            # [batch_size]
            loss = tf.reduce_sum(per_example_loss)

        # [batch_size, seq_length, num_labels]
        probabilities = tf.nn.softmax(logits, axis=-1)
        # [batch_size, seq_length] , name = 'loss/ArgMax'
        predict = tf.argmax(probabilities, axis=-1)

        return (loss, per_example_loss, logits, predict)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_one_hot_embeddings):
    """
    构建模型, 把model_fn传递到Estimator中，因为Estimator不接受带参数的model_fn
    :arg
    bert_config: bert超参数配置文件的实例
    num_labels: 标签的数量，这里不同的是需要对导入的标签 +1，因为有[PAD]未在内
    init_checkpoint: 预训练模型，这里是~/chinese_L-12_H-768_A-12
    learning_rate: 学习率
    num_train_steps: 训练步数
    num_warmup_steps: 学习率衰减步数
    use_one_hot_embeddings: 是否使用ne_hot编码embedding
    """
    def model_fn(features, labels, mode, params):
        """
        模型，包括载入预训练模型、构建bert模型、构建模型输出层
        :arg
        features: example
        """
        # 打印传入数据的信息
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = None
        if FLAGS.do_train or FLAGS.do_eval or FLAGS.do_predict:
            label_ids = features["label_ids"]

        # 判断是否是训练
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, predicts) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        tf.logging.info('total_loss{}:'.format(total_loss))

        tvars = tf.trainable_variables()
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            # 使用指定初始化方式，使用init_checkpoint中的变量初始化当前变量
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            # 答应变量的信息
            # tf.logging.info("**** Trainable Variables ****")
            # for var in tvars:
            #     init_string = ""
            #     if var.name in initialized_variable_names:
            #         init_string = ", *INIT_FROM_CKPT*"
            #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, False)
            hook_dict = dict()
            hook_dict['loss'] = total_loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=50)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, pred_ids):
                label_ids = tf.cast(label_ids, tf.int32)
                pred_ids = tf.cast(pred_ids, tf.int32)
                return {
                    "eval_loss": tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids)
                }

            eval_metrics = metric_fn(label_ids, predicts)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
            )

        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predicts
            )

        if FLAGS.do_inference:
            tf.logging.info('****** inference *******')
            predictions_dict = {"predictions": predicts}
            export_outputs = {
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    tf.estimator.export.PredictOutput(predictions_dict)}
            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     predictions=predictions_dict,
                                                     export_outputs=export_outputs)

        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    task_name = FLAGS.task_name.lower()
    # 检查output_dir是否存在
    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    processors = {
        'ner': NerProcessor
    }

    # 验证是否模型具有处理某一任务的能力
    if task_name not in processors:
        raise ValueError('Task not found: %s' % task_name)
    else:
        tf.logging.info('start %s task' % task_name)

    # if not FLAGS.do_train and not FLAGS.do_eval:
    #     raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # 读取配置bert超参数的配置文件
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    # max_seq_length参数涉及到微调和测试输入的序列长度，他是不能超过已经指定的最大位置的
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    # 实例化处理器
    processor = processors[task_name]()
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    # 标签类型列表, 例如 ['B-COM', 'B-PER', 'I-PER', ...]
    label_list = processor.get_labels()

    # 设置session的配置
    sess_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True
    )

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_summary_steps=FLAGS.save_summary_steps,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        session_config=sess_config
    )

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    # 训练过程如下：
    if FLAGS.do_train:
        # 1. 加载数据，并打印出训练集的详细信息
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        if num_train_steps < 1:
            raise AttributeError('training data is so small...')
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

    # 2. 构造模型网络，用于Estimator的model_fn
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

    params = {
        'batch_size': FLAGS.train_batch_size
    }

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)

    if FLAGS.do_train:
        # 3. 把训练样本写入TF_Record中,当已经存在了TFRcord文件的时候，跳过此步骤
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if not tf.gfile.Exists(train_file):
            filed_based_convert_examples_to_features(
                train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, FLAGS.output_dir)

        # 4. 读取TFRecord文件中的训练数据，用于Estimator的input_fn
        train_input_fn = file_based_input_fn_builder(input_file=train_file,
                                                     seq_length=FLAGS.max_seq_length,
                                                     is_training=True,
                                                     drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, FLAGS.output_dir)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=True)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        with open('./{}/label2id.pkl'.format('checkpoint'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                 FLAGS.max_seq_length, tokenizer,
                                                 predict_file, FLAGS.output_dir,
                                                 mode="test")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "label_test1.txt")
        with open(output_predict_file, 'w') as writer:
            for prediction in result:
                output_line = "\n".join(id2label[i] for i in prediction if i != 0) + "\n"
                writer.write(output_line)

    if FLAGS.do_inference:
        estimator.export_savedmodel(FLAGS.export_dir_base, serving_input_receiver_fn,
                                    strip_default_attrs=True)


if __name__ == '__main__':
    tf.app.run()
