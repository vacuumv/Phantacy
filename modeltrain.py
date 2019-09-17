import json

import pandas as pd
import jieba
import jieba.analyse
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
import random
from sklearn.svm import SVC, LinearSVC
from IPython.display import display

classifiers = [
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # AdaBoostClassifier(),
    MultinomialNB(alpha=2)
]

jieba.set_dictionary('dict.txt.big')

with open('stop_words.txt', encoding='utf-8') as stop_file:
    stopwords = stop_file.readlines()
stopwords = [w.strip() for w in stopwords]


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class PostWriterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dummy_train = None

    def fit(self, x, y=None, **fit_params):
        print("萃取貼文者ID特徵向量中")
        self.dummy_train = pd.get_dummies(x)

    def transform(self, x):
        print("將貼文者ID轉換為向量中")
        dummy_new = pd.get_dummies(x)
        return dummy_new.reindex(columns=self.dummy_train.columns, fill_value=0)

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y, **fit_params)
        return self.transform(x)


class TimeIntervalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dummy_train = None

    def fit(self, x, y=None, **fit_params):
        print("萃取時間間隔向量中")
        # 轉換成小時
        new_x = []
        for time in x:
            new_x.append(time / 60 / 60 if time is not None else 0)
        self.dummy_train = pd.get_dummies(new_x)

    def transform(self, x):
        print("時間間隔轉換為向量中")
        new_x = []
        for time in x:
            new_x.append(time / 60 / 60 if time is not None else 0)
        dummy_new = pd.get_dummies(new_x)
        return dummy_new.reindex(columns=self.dummy_train.columns, fill_value=0)

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y, **fit_params)
        return self.transform(x)


class PostReviewTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dummy_train = None

    def fit(self, x, y=None, **fit_params):
        print("萃取回文者ID特徵向量中")
        new_x = {}
        for iid, item in enumerate(x):
            new_x[iid] = ["0"] if len(item) == 0 else item
        x = pd.Series(new_x)
        self.dummy_train = pd.get_dummies(x.apply(pd.Series).stack()).sum(level=0)

    def transform(self, x):
        print("將回文者ID轉換為向量中")
        new_x = {}
        for iid, item in enumerate(x):
            new_x[iid] = ["0"] if len(item) == 0 else item
        x = pd.Series(new_x)
        dummy_new = pd.get_dummies(x.apply(pd.Series).stack()).sum(level=0)
        return dummy_new.reindex(columns=self.dummy_train.columns, fill_value=0)

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y, **fit_params)
        return self.transform(x)


class ModelGenerator:
    def __init__(self, output_file_path):
        self.is_complete = False
        self.result_location = output_file_path
        self.classifier = None

    @staticmethod
    def _read_result(file_name):
        with open(file_name, encoding='utf-8') as json_data:
            posts = json.load(json_data)
            return posts

    @staticmethod
    def _transform_input_matrix(data, partition=1.0):
        """
        轉換原始資料成為sklearn可以分析的格式，切成兩等份或不切等分
        :param data: 原始資料
        :param partition: 切等分之比例 1為不切回傳一整份, 例如0.8則為回傳0.8, 0.2的資料
        :return: sklearn接受的格式資料
        """
        print("轉換資料至訓練模型之輸入格式")
        total_size = len(data)
        main_size = int(total_size * partition)
        minor_size = total_size - main_size
        keys = data[0].keys()
        part1 = {}
        random.shuffle(data)
        for key in keys:
            part1[key] = [post[key] for post in data[:main_size]]
        if partition == 1:
            print("轉換完畢")
            return pd.DataFrame(part1)
        else:
            part2 = {}
            for key in keys:
                part2[key] = [post[key] for post in data[-minor_size:]]
            return pd.DataFrame(part1), pd.DataFrame(part2)

    @staticmethod
    def _get_training_pipeline(post_content, post_comment_ids, post_writer_ids, time_interval, classifier):
        tokenizer = lambda sentence: [token for token in jieba.cut(sentence) if token not in stopwords]
        transformer_list = []
        transformer_weights = {}

        if post_content > 0:
            transformer_list.append(('post_content', Pipeline([
                ('selector', ItemSelector(key='post_content')),
                ('vect', CountVectorizer(tokenizer=tokenizer)),
                ('tfidf', TfidfTransformer()),
            ])))
            transformer_weights['post_content'] = post_content

        if post_comment_ids > 0:
            transformer_list.append(('post_comment_ids', Pipeline([
                ('selector', ItemSelector(key='post_comment_ids')),
                ('vec_trans', PostReviewTransformer()),
            ])))
            transformer_weights['post_comment_ids'] = post_comment_ids

        if post_writer_ids > 0:
            transformer_list.append(('post_writer_id', Pipeline([
                ('selector', ItemSelector(key='post_writer_id')),
                ('vec_trans', PostWriterTransformer()),
            ])))
        transformer_weights['post_writer_id'] = post_comment_ids

        if time_interval > 0:
            transformer_list.append(('time_interval', Pipeline([
                ('selector', ItemSelector(key='time_interval')),
                ('ver_trans', TimeIntervalTransformer()),
            ])))
        transformer_weights['time_interval'] = time_interval

        clf = Pipeline([
            ('union', FeatureUnion(transformer_list=transformer_list, transformer_weights=transformer_weights)),
            ('clf', classifier)
        ])

        return clf

    def predict(self, content, reply=None):
        print("模型預測資訊：")
        if self.classifier is None:
            print("模型不存在請先訓練")
            return
        predicted = self.classifier.predict(content)
        print("留言：")
        if reply is not None:
            output_dict = {
                "是否該回應": predicted,
                "實際數據": reply,
                "留言": [str(content).replace(" ", "") for content in content['post_content']]
            }
            data_frame = pd.DataFrame(output_dict)
            display(data_frame)
            rate = np.mean(predicted == reply)
            print("準確率: {0:.2f}%".format(rate * 100))
            return data_frame
        else:
            output_dict = {
                "是否該回應": predicted,
                "留言": [content.strip() for content in content['post_content']]
            }
            data_frame = pd.DataFrame(output_dict)
            display(data_frame)
            return data_frame

    @staticmethod
    def store_predict_output(data_frame, file_name):
        data_frame.to_csv(file_name, sep='\t', encoding='utf-8')

    def train(self, post_content, post_comment_ids, post_writer_ids, time_interval):
        posts = self._read_result(self.result_location)
        partition = 0.8
        training_data_set, testing_data_set = self._transform_input_matrix(data=posts, partition=partition)
        total_size = len(posts)
        training_size = int(total_size * partition)
        testing_size = total_size - training_size
        print("訓練資料總大小: {}筆".format(total_size))
        print("訓練集大小: {}".format(training_size))
        print("測試集大小: {}".format(testing_size))

        testing_replies = testing_data_set['fan_page_reply']
        train_replies = training_data_set['fan_page_reply']

        print("模型訓練中 請耐心等待")
        for classifier in classifiers:
            print(str(classifier))
            self.classifier = self._get_training_pipeline(post_content, post_comment_ids,
                                                          post_writer_ids, time_interval
                                                          , classifier)
            self.classifier.fit(training_data_set, train_replies)
            self.predict(testing_data_set, testing_replies)


def main():
    # 訓練資料來源
    output_file_path = "/Users/Steve/PycharmProjects/Phantacy/result_all.json"
    post_content_weight = 1.5
    post_comment_ids_weight = 1.0
    post_writer_ids_weight = 1.0
    time_interval_weight = 1.0
    model_generator = ModelGenerator(output_file_path=output_file_path)

    model_generator.train(post_content=post_content_weight,
                          post_comment_ids=post_comment_ids_weight,
                          post_writer_ids=post_writer_ids_weight,
                          time_interval=time_interval_weight)


if __name__ == '__main__':
    main()
