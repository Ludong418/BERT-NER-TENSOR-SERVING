#!/usr/bin/python

# encoding: utf-8

"""
@author: dong.lu

@contact: ludong@cetccity.com

@software: PyCharm

@file: utils_.py

@time: 2019/04/9 10:30

@desc: 处理句子
"""

import re


class SentenceProcessor(object):
    def __init__(self):
        self.sentence_index = 0

    @staticmethod
    def cut_sentence(sentence):
        """
        分句
        :arg
        sentence: string类型，一个需要分句的句子

        :return
        返回一个分好句的列表
        """
        sentence = re.sub('([。！？\?])([^”’])', r"\1\n\2", sentence)
        sentence = re.sub('(\.{6})([^”’])', r"\1\n\2", sentence)
        sentence = re.sub('(\…{2})([^”’])', r"\1\n\2", sentence)
        sentence = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', sentence)
        sentence = sentence.rstrip()

        return sentence.split("\n")

    def concat_sentences(self, sentences, max_seq_length):
        """
        把一个列表里的句子按照小于最大长度拼接为另一个句子，当某几个句子拼接
        达到max_seq_length长度的时候，把这个新的句子保存到新的列表当中。

        :arg
        sentences: list类型，一个要别拼接的句子列表
        max_seq_length: 拼接的新句子的最大长度

        :return
        一个新的拼接句子的列表，元素为string类型，即句子
        """
        # 分句后并从新拼接的句子
        new_sentences = []
        # 句子的index，是一个两层的列表，例如 [[0], [1, 2]], 列表内的每一个列表，
        # 表示来源于同一个句子，这里[1, 2]就表示是同一个句子被分割的两个句子
        sentences_index = []

        for sentence in sentences:
            sentence = self.clean_sentence(sentence)
            # 如果句子小于且等于最大长度的话，不进行处理
            if len(sentence) <= max_seq_length:
                new_sentences.append(sentence)
                sentences_index.append([self.sentence_index])
                self.sentence_index += 1

            # 如果句子大于最大长度就需要进行切割句子再拼接的操作了
            else:
                # 产生拼接句子列表（列表内每个句子小于最大长度）和同一个句子的index列表
                single_sentences, singe_index = self.concat_single_sentence(sentence, max_seq_length)
                new_sentences.extend(single_sentences)
                sentences_index.append(singe_index)

        # 当调用完此函数后，需要把sentence_index设为0，否则下次再次使用时候，将不会从0开始记录
        self.sentence_index = 0

        return new_sentences, sentences_index

    def concat_single_sentence(self, sentence, max_seq_length):
        """
        把一个句子分句为多个句子，把这些句子再拼接成若干个小于
        max_seq_length的句子

        :arg
        sentence: string类型，待分割的句子

        :return
        拼接后的句子列表和同一个句子的index列表
        """
        # 拼接后的句子列表
        single_sentences = []
        # 同一个句子的index列表
        singe_index = []
        tmp = ''
        # 分句， 注意此时sentence为list类型
        sentence = self.cut_sentence(sentence)
        for i, sent in enumerate(sentence):
            tmp = tmp + sent
            if len(tmp) > max_seq_length:
                pre = tmp[0: len(tmp) - len(sent)]
                if len(pre) >= 2:
                    single_sentences.append(pre)
                    singe_index.append(self.sentence_index)
                    self.sentence_index += 1
                tmp = sent

            # 当遍历到最后一个的时候，且tmp不为空字符串，就把tmp存入single_sentences中
            if i == len(sentence) - 1 and len(tmp) >= 2:
                single_sentences.append(tmp)
                singe_index.append(self.sentence_index)
                self.sentence_index += 1

        return single_sentences, singe_index

    @staticmethod
    def clean_sentence(sentence):
        sentence = sentence.strip()
        sentence = re.sub('\t| ', '', sentence)

        return sentence


if __name__ == '__main__':
    test = ['四天三次!4月14日，马云第三次谈996，他表示看到了大家的质疑，但还是想说实话。“12315,996关键在于找到自己喜欢的事，真正的996不是简单加班，而是把时间用在学习和提升自己，爱觉不累，但企业不能不给钱。',
        '马云老师的一番言论，又在网上引起热议。',
        '我是',
        '你知不知道龙门石窟在哪个地方']
    sp = SentenceProcessor()
    a, b = sp.concat_sentences(test, 62)