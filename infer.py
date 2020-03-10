#!/usr/bin/python

# encoding: utf-8

"""
@author: dong.lu

@contact: ludong@cetccity.com

@software: PyCharm

@file: infer.py

@time: 2019/04/9 10:30

@desc: 模型推理部分，分为本地载入模型推理或者tensorflow serving grpc 推理

"""
import os
import grpc
import codecs
import pickle
import warnings

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import types_pb2

from bert import tokenization
import utils_


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class Entity(object):
    def __init__(self, types):
        self.__begin = None
        self.types = types
        self.__intermediate = []

    @property
    def intermediate(self):
        return self.__intermediate

    @intermediate.setter
    def intermediate(self, intermediate):
        self.__intermediate.append(intermediate)

    @property
    def begin(self):
        return self.__begin

    @begin.setter
    def begin(self, begin):
        self.__begin = begin

    def get_entity_types(self):
        return self.__begin + ''.join(self.__intermediate), self.types


class InferenceBase(object):
    def __init__(self, vocab_file, labels, url=None, model_name=None,
                 signature_name=None, export_dir=None, do_lower_case=True):
        """
        预测的基类，分为两种方式预测
            a. grpc 请求方式
            b. 本地导入模型方式

        :arg
        vocab_file: bert 预训练词典的地址，这里在 'chinese_L-12_H-768_A-12/vocab.txt '中
        labels: str 或 list 类型，需要被转化为id的label，当为str类型的时候，即为标签-id的pkl文件名称；
                当为list时候，即为标签列表。
        url: string类型，用于调用模型测试接口，host:port，例如'10.0.10.69:8500'
        export_dir: string类型，模型本地文件夹目录，r'model\1554373222'
        model_name: string类型，tensorflow serving 启动的时候赋予模型的名称，当
                    url被设置的时候一定要设置。
        signature_name: string类型，tensorflow serving 的签名名称，当
                    url被设置的时候一定要设置。
        do_lower_case: 是否进行小写处理

        :raise
        url和export_dir至少选择一个，当选择url的时候，model_name和signature_name不能为
        None。
        """
        self.url = url
        self.export_dir = export_dir

        if export_dir:
            self.predict_fn = tf.contrib.predictor.from_saved_model(self.export_dir)

        if self.url:
            channel = grpc.insecure_channel(self.url)
            self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
            self.request = predict_pb2.PredictRequest()
            self.model_name = model_name
            self.signature_name = signature_name

            self.request.model_spec.name = self.model_name

            self.request.model_spec.signature_name = self.signature_name

            if self.model_name is None or self.signature_name is None:
                raise ValueError('`model_name` and `signature_name` should  not NoneType')

        if url is None and export_dir is None:
            raise ValueError('`url` or `export_dir`is at least of one !')

        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.id2label = self._load_id_map_label(labels)

    def local_infer(self, examples):
        """
        导入本地的PB文件进行预测
        """
        pass

    def tf_serving_infer(self, examples):
        """
        使用tensorflow serving进行grpc请求预测
        """
        pass

    def preprocess(self, sentences, max_seq_length):
        pass

    def create_example(self):
        pass

    @staticmethod
    def _load_id_map_label(labels=None):
        id2label = {}
        if isinstance(labels, list):
            for i, label in enumerate(labels, 1):
                id2label[i] = labels

        elif isinstance(labels, str):
            with codecs.open(labels, 'rb') as rf:
                label2id = pickle.load(rf)
                id2label = {value: key for key, value in label2id.items()}

        return id2label


class NerInfer(InferenceBase):
    def __init__(self, vocab_file, labels, url=None, model_name=None,
                 signature_name=None, export_dir=None, do_lower_case=True):
        """
        bert ner, 参数解释查看 `InferenceBase`
        """
        super(NerInfer, self).__init__(vocab_file, labels, url, model_name, signature_name, export_dir, do_lower_case)
        self.sentenceprocessor = utils_.SentenceProcessor()

    def preprocess(self, sentences, max_seq_length):
        """
        对sentences进行预处理，并生成examples

        :arg
        sentences: 二维列表，即输入的句子，输入有一下要求：
                （1）可以是一段话，但是每个句子最好小于64个字符串长度
                （2）长度不可以小于2
        max_seq_length: 输入的每一个句子的最大长度

        :return
        examples: tf.train.Example对象
        new_tokens: 二维列表，sentences清洗后的tokens
        sentences_index: 二维列表，分句后，对应到原始句子的下标
                        例如：[[0], [1, 2]...]
        """
        if not sentences or not isinstance(sentences, list):
            raise ValueError('`sentences` must be list object and not a empty list !')

        # 把sentences中的句子是不是小于或等于max_seq_length的，进行分句，然后在拼接成若干个
        # 小于或等于max_seq_length的的句子
        new_sentences, sentences_index = self.sentenceprocessor.concat_sentences(sentences, max_seq_length)

        examples, new_tokens = [], []
        for sentence in new_sentences:
            feature, ntokens = self.convert_single_example(sentence, max_seq_length)
            features = dict()
            features['input_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature.input_ids))
            features['input_mask'] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature.input_mask))
            features['segment_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature.segment_ids))
            example = tf.train.Example(features=tf.train.Features(feature=features))
            examples.append(example.SerializeToString())
            new_tokens.append(ntokens)

        return examples, new_tokens, sentences_index

    def convert_single_example(self, sentence, max_seq_length):
        """
        对单个句子进行token、转id、padding等处理

        :arg
        sentence: string类型，单个句子
        max_seq_length: 句子的最大长度

        :return
        feature: InputFeatures对象
        ntokens: 处理句子后得到的token
        """
        tokens = self.tokenizer.tokenize(sentence)

        # 序列截断
        if len(tokens) >= max_seq_length - 1:
            # -2 的原因是因为序列需要加一个句首和句尾标志
            tokens = tokens[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []

        ntokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens:
            ntokens.append(token)
            segment_ids.append(0)

        # 句尾添加[SEP]标志
        ntokens.append("[SEP]")
        segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)

        # padding
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # ntokens.append("**NULL**")

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # 结构化为一个类
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids
        )

        return feature, ntokens[1: -1]

    def infer(self, sentences, max_seq_length):
        """
        对外的测试接口

        :arg
        sentences:  二维列表，即输入的句子，输入有一下要求：
                （1）可以是一段话，但是每个句子最好小于64个字符串长度
                （2）长度不可以小于2
        max_seq_length: 输入的每一个句子的最大长度

        sentences_entities: 返回每一个句子的实体
        """
        examples, new_tokens, sentences_index = self.preprocess(sentences, max_seq_length)
        if self.url:
            predictions = self.tf_serving_infer(examples)

        else:
            predictions = self.local_infer(examples)

        result = self.convert_id_to_label(predictions['predictions'])

        # debug
        # for t, r in zip(new_tokens, result):
        #     if len(r) != len(t):
        #         warnings.warn('Token and tags have different lengths.\ndetails:\n{}\n{}'.format(t, r))
        #     print(list(zip(t, r)))

        sentences_entities = self.get_entity(new_tokens, result, sentences_index)

        return sentences_entities

    def tf_serving_infer(self, examples):
        """
        使用tensorflow serving预测

        :arg
        examples: tf.train.Example 对象

        :return
        二维列表，预测结果
        """
        self.request.inputs['examples'].CopyFrom(tf.make_tensor_proto(examples, dtype=types_pb2.DT_STRING))
        response = self.stub.Predict(self.request, 5.0)

        predictions = {}
        for key in response.outputs:
            tensor_proto = response.outputs[key]
            nd_array = tf.contrib.util.make_ndarray(tensor_proto)
            predictions[key] = nd_array

        return predictions

    def local_infer(self, examples):
        """
        本地进行预测，参数解释同上
        """
        predictions = self.predict_fn({'examples': examples})

        return predictions

    def convert_id_to_label(self, predictions):
        """
        把预测结果变为label

        :arg
        predictions: 二维列表，测试结果，[[1,2,3], [2,3,4]...]

        :return
        result: 二维列表，转变的结果
        """
        result = []
        for prediction in predictions:
            curr_result = []
            for idx in prediction:
                if idx == 0:
                    break

                curr_label = self.id2label[idx]
                if curr_label in ['[CLS]', '[SEP]']:
                    continue
                curr_result.append(curr_label)
            result.append(curr_result)

        return result

    @staticmethod
    def get_entity(tokens, tags, sentences_index):
        """
        提取实体

        :arg
        tokens: 二维列表，句子处理后得到的token
        tags: 二维列表，预测的结果
        sentences_index: 二维列表，句子拆分后，对应到原句的index

        :return
        sentences_entities: 二维列表，返回实体结果，例如[('昆凌', 'PER')...]
        """
        sentences_entities = []
        for sent in sentences_index:
            entities = []
            for i in sent:
                if len(tokens[i]) != len(tags[i]):
                    warnings.warn('Token and tags have different lengths.\ndetails:\n{}\n{}'.format(tokens[i], tags[i]))

                entity = Entity(None)
                t_zip = zip(tokens[i], tags[i])

                for token, tag in t_zip:
                    if tag == 'O':
                        if entity.types:
                            entities.append(entity.get_entity_types())
                            entity = Entity(None)
                        continue

                    elif tag[0] == 'B':
                        if entity.types:
                            entities.append(entity.get_entity_types())
                        entity = Entity(tag[2:])
                        entity.begin = token

                    elif tag[0] == 'I':
                        try:
                            entity.intermediate = token
                        except Exception as e:
                            print(e)

            sentences_entities.append(entities)

        return sentences_entities


if __name__ == '__main__':
    project_path = os.path.dirname(os.path.abspath(__file__))
    export_dir = project_path + os.sep + 'model' + os.sep + '1554373222'
    vocab_file = project_path + '{}chinese_L-12_H-768_A-12{}vocab.txt'.format(os.sep, os.sep)
    labels = project_path + '{}checkpoint{}label2id.pkl'.format(os.sep, os.sep)
    # url = '10.0.10.69:8500'
    # export_dir = r'E:\ludong\bert_ner\model\1554373222'
    nerinfer = NerInfer(vocab_file, labels, url='10.0.10.69:8500', model_name='models', signature_name='serving_default')
    # entities = nerinfer.infer(utils_.test, 64)
    # print(entities)

    while True:
        sentence = input('请输入句子：')
        print(nerinfer.infer([sentence], 64))
