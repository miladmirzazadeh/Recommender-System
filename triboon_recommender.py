#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[8]:


import implicit
import json
import collections
import os
import pickle

import numpy as np
from scipy.sparse import lil_matrix


import logging

logger = logging.getLogger(__name__)



class CFUtils:

    @staticmethod
    def get_cf_base_dir(_key: str):
        return os.path.join('./', f'cf/{_key}/')

    @staticmethod
    def delete_website_cf_models(_key: str):
        try:
            shutil.rmtree(CFUtils.get_cf_base_dir(_key=_key))
        except FileNotFoundError:
            pass



class ViewMatrix:
    def __init__(self, key):
        self._key = key
        self._original = True
        self._view_matrix = lil_matrix((0, 0), dtype=np.float64)
        self._item_indexer = AppendIndexer()
        self._user_indexer = AppendIndexer()

    @staticmethod
    def load_matrix(key: str):
        matrix = ViewMatrix(key)

        view_matrix_dir = os.path.join(CFUtils.get_cf_base_dir(_key=key), 'view_matrix')
        try:
            matrix._view_matrix = ViewMatrix                 .load_sparse_lil(os.path.join(view_matrix_dir, 'lil_matrix.npz'))
        except Exception as e:
            logger.warning('error happened while loading ViewMatrix for site: %s, reason: %s', key, e)

        try:
            matrix._item_indexer = AppendIndexer.load(
                os.path.join(view_matrix_dir, 'item_indexer.indexer'))
        except Exception as e:
            logger.warning('error happened while loading item_indexer for site: %s, reason: %s', key, e)

        try:
            matrix._user_indexer = AppendIndexer.load(
                os.path.join(view_matrix_dir, 'user_indexer.indexer'))
        except Exception as e:
            logger.warning('error happened while loading user_indexer for site: %s, reason: %s', key, e)

        return matrix

    def to_csr(self):
        train_data = self._view_matrix.astype(np.float64)
        train_data = train_data.tocoo()
        train_data.data = np.log10(train_data.data) + 1
        train_data = train_data.tocsr()
        return train_data

    @staticmethod
    def save_sparse_lil(filename, array):
        np.savez_compressed(filename, dtype=array.dtype.str, data=array.data, rows=array.rows, shape=array.shape)

    @staticmethod
    def load_sparse_lil(filename):
        loader = np.load(filename, allow_pickle=True)
        result = lil_matrix(tuple(loader["shape"]), dtype=str(loader["dtype"]))
        result.data = loader["data"]
        result.rows = loader["rows"]
        return result

    def remained_indices(removing_indices, max_index):
        return list(set(range(max_index)) - set(removing_indices))

    def make_dense(self, user_min_view, item_min_view):
        udm_csc = self._view_matrix.tocsc()
        removing_column_indices = list(np.where(udm_csc.getnnz(0) <= item_min_view)[0])
        self._item_indexer.remove_indexes(removing_column_indices)
        remaining_indices = ViewMatrix.remained_indices(removing_column_indices, self._view_matrix.shape[1])
        udm_csc = udm_csc[:, remaining_indices]

        udm_csc = udm_csc.tocsr()
        removing_row_indices = list(np.where(udm_csc.getnnz(1) <= user_min_view)[0])
        self._user_indexer.remove_indexes(removing_row_indices)
        remaining_indices = ViewMatrix.remained_indices(removing_row_indices, self._view_matrix.shape[0])
        self._view_matrix = udm_csc[remaining_indices, :]

    # @log_enter_exit
    # def make_dense(self, user_min_view, item_min_view):
    #     self._original = False
    #     while True:
    #         removed_rows_cnt = self.trim_users_with_few_views(user_min_view)
    #         removed_columns_cnt = self.trim_columns_with_few_views(item_min_view)
    #         if not removed_columns_cnt and not removed_rows_cnt:
    #             break

    def trim_users_with_few_views(self, user_min_view):
        removing_row_indices = list(np.where(self._view_matrix.getnnz(1) < user_min_view)[0])
        logger.info('Number of users which should be deleted: %d', len(removing_row_indices))
        self.trim_user_indices(to_remove_indices=removing_row_indices)
        return len(removing_row_indices)

    def trim_columns_with_few_views(self, column_min_view):
        removing_column_indices = list(np.where(self._view_matrix.getnnz(0) < column_min_view)[0])
        logger.info('number products which should be deleted: %d', len(removing_column_indices))
        self.trim_column_indices(to_remove_indices=removing_column_indices)
        return len(removing_column_indices)

    def trim_user_indices(self, to_remove_indices):
        self._user_indexer.remove_indexes(to_remove_indices)
        self._view_matrix = ViewMatrix.delete_row_lil(self._view_matrix, to_remove_indices)

    def trim_column_indices(self, to_remove_indices):
        self._item_indexer.remove_indexes(to_remove_indices)
        self._view_matrix = ViewMatrix.delete_column_lil(self._view_matrix, to_remove_indices)

    def delete_column_lil(mat: lil_matrix, *i) -> lil_matrix:
        mat = mat.transpose()
        mat = ViewMatrix.delete_row_lil(mat, *i)
        return mat.transpose()

    def delete_row_lil(mat: lil_matrix, *i) -> lil_matrix:
        if not isinstance(mat, lil_matrix):
            raise ValueError("works only for LIL format -- use .tolil() first")
        mat = mat.copy()
        mat.rows = np.delete(mat.rows, i)
        mat.data = np.delete(mat.data, i)
        mat._shape = (mat.rows.shape[0], mat._shape[1])
        return mat

    def save(self):
        if not self._original:
            raise Exception('this matrix should not be saved. it may have missing data')

        redis_connection = RedisClients.get_redis_connection('default')
        try:
            with Lock(redis=redis_connection, blocking=False, blocking_timeout=0,
                      name=f'cf:matrix:{self._key}', timeout=900):

                view_matrix_dir = os.path.join(CFUtils.get_cf_base_dir(_key=self._key), 'view_matrix/')
                create_folder(view_matrix_dir)

                self.save_sparse_lil(os.path.join(view_matrix_dir, 'lil_matrix.npz'), self._view_matrix)
                self._item_indexer.dump(os.path.join(view_matrix_dir, 'item_indexer.indexer'))
                self._user_indexer.dump(os.path.join(view_matrix_dir, 'user_indexer.indexer'))
        except LockError:
            logger.info('Could not get lock to save indexers for matrix_key %s', self._key)
            raise LockAcquireError

    def add_value(self, user_token, page_id, value):
        user_index = self._user_indexer.get_or_create(user_token)
        item_index = self._item_indexer.get_or_create(page_id)
        d1, d2 = self._view_matrix.shape
        if user_index >= d1:
            d1 = (d1 + 1) * 2
        if item_index >= d2:
            d2 = (d2 + 1) * 2
        if d1 > self._view_matrix.shape[0] or d2 > self._view_matrix.shape[1]:
            self._view_matrix.resize(d1, d2)
        self._view_matrix[user_index, item_index] += int(value)

    def shape(self):
        return self._item_indexer.size, self._user_indexer.size



    
def create_folder(_path):
    directory = os.path.dirname(_path)
    os.makedirs(directory, exist_ok=True)


class AppendIndexer(object):

    def __init__(self):
        self.__indices = {}
        self.__indices_reverse = {}
        self.__next_idx = 0

    @classmethod
    def load(cls, filename):
        obj = cls()
        with open(filename, 'rb') as fin:
            obj.__indices, obj.__indices_reverse, obj.__next_idx = pickle.load(fin)
        return obj

    def dump(self, filename):

        tmp_filename = filename + '.tmp'
        create_folder(filename)

        with open(tmp_filename, 'wb') as fout:
            pickle.dump((self.__indices, self.__indices_reverse, self.__next_idx), fout)

        os.rename(tmp_filename, filename)

    def is_in(self, item: str) -> bool:
        return item in self.__indices

    def get(self, item: object) -> object:
        return self.__indices[str(item)]

    def reverse_get(self, index: int) -> str:
        return self.__indices_reverse.get(index)

    def remove_index(self, index: int):
        return self.remove_indexes([index])

    def remove_indexes(self, indexes: [int], get_new=0):
        indexes = set(filter(lambda idx: idx < self.size, indexes))

        if len(indexes) == 0:
            return self

        new_indexer = AppendIndexer()
        for i in range(self.size):
            if i not in indexes:
                new_indexer.get_or_create(self.reverse_get(i))

        if get_new:
            return new_indexer

        self.__indices = new_indexer.__indices
        self.__indices_reverse = new_indexer.__indices_reverse
        self.__next_idx = new_indexer.__next_idx

        return self

    def get_or_create(self, item) -> int:
        item = str(item)
        if item not in self.__indices:
            self.__indices[item] = idx = self.__next_idx
            self.__indices_reverse[idx] = item
            self.__next_idx += 1
        else:
            idx = self.__indices[item]
        return idx

    def get_items(self):
        return self.__indices.keys()

    @property
    def size(self) -> int:
        return self.__next_idx


# In[19]:


import numpy as np
from implicit import _als


class Als():

    def __init__(self,
                 num_factors=40,
                 regularization=0.01,
                 alpha=1.0,
                 iterations=15,
                 use_native=True,
                 num_threads=0,
                 dtype=np.float64):
        """
        Class version of alternating least squares implicit matrix factorization
        Args:
            num_factors (int): Number of factors to extract
            regularization (double): Regularization parameter to use
            iterations (int): Number of alternating least squares iterations to
            run
            use_native (bool): Whether or not to use Cython solver
            num_threads (int): Number of threads to run least squares iterations.
            0 means to use all CPU cores.
            dtype (np dtype): Datatype for numpy arrays
        """
        self.num_factors = num_factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.use_native = use_native
        self.num_threads = num_threads
        self.dtype = dtype

        self.user_vectors = None
        self.item_vectors = None
        self.solver = None

    def fit(self, Cui):
        """
        Fit an alternating least squares model on Cui data
        Args:
            Cui (sparse matrix, shape=(num_users, num_items)): Matrix of
            user-item "interactions"
        """

        users, items = Cui.shape

        self.user_vectors = np.random.normal(size=(users, self.num_factors))             .astype(self.dtype)

        self.item_vectors = np.random.normal(size=(items, self.num_factors))             .astype(self.dtype)

        self.solver = _als.least_squares
        self.fit_partial(Cui)

    def fit_partial(self, Cui):
        """Continue fitting model"""
        # scaling
        Cui = Cui.copy()
        Cui.data *= self.alpha
        Cui, Ciu = Cui.tocsr(), Cui.T.tocsr()

        for iteration in range(self.iterations):
            self.solver(Cui,
                        self.user_vectors,
                        self.item_vectors,
                        self.regularization,
                        self.num_threads)
            self.solver(Ciu,
                        self.item_vectors,
                        self.user_vectors,
                        self.regularization,
                        self.num_threads)

    def predict(self, user, item):
        """Predict for single user and item"""
        return self.user_vectors[user, :].dot(self.item_vectors[item, :].T)

    def predict_for_customers(self, ):
        """Recommend products for all customers"""
        return self.user_vectors.dot(self.item_vectors.T)

    def predict_for_items(self, norm=True):
        """Recommend products for all products"""
        pred = self.item_vectors.dot(self.item_vectors.T)
        if norm:
            norms = np.array([np.sqrt(np.diagonal(pred))])
            pred = pred / norms / norms.T
        return pred


# In[ ]:





# In[21]:


class Recommender():
    '''
    alpha value must be determined based on the users' rationality. 
    If users choose the plans with very careful analysis, alpha should be high (close to 100). otherwise, it should be decreased to zero.
    '''
    def __init__(self, alpha=40):
        self.matrix = ViewMatrix('triboon_matrix')
        self.alpha = alpha
        self.als_model = Als(num_factors=10,
                iterations=20,
                num_threads=1,
                alpha=self.alpha)
    def train_new_model(self, purchased_items:dict):
        for user, plans in purchased_items.items():
            for plan_id in plans:
                self.matrix.add_value(user, plan_id, 1)
        self.matrix.make_dense(user_min_view=2, item_min_view=5)

        
        train_data = self.matrix.to_csr()
        self.als_model.fit(train_data)
    
    def predict_all_users(self):
        final_arr = self.als_model.user_vectors.dot(self.als_model.item_vectors.T)
        final_dict= dict()
        for i in range(self.matrix._user_indexer.size):
            final_dict[self.matrix._user_indexer.reverse_get(i)] = self.weights_to_ranks(final_arr[i])
        return(final_dict)
    
    def predict_batch_of_users(self,purchased_items:dict):
        final_dict = dict()
        for user, items in purchased_items.items():
            final_dict[user] = self.recommend_new_items(items)
        return(final_dict)
    #         return all users recommendation

    def recommend_new_items(self, previous_items):
        items_vec = self.items_to_vec(previous_items)
        o = items_vec.dot(self.als_model.item_vectors)
        predict = o.dot(self.als_model.item_vectors.T)
        final_list = self.weights_to_ranks(predict)
        return(final_list)
        
    def items_to_vec(self, items):
        arr = np.zeros(self.matrix._item_indexer.size)
        for i in items:
            print(int(self.matrix._item_indexer.get(i)))
            arr[self.matrix._item_indexer.get(i)] += 1
            
        return(arr)
    
    def weights_to_ranks(self, weights):
        dd = dict(zip(self.matrix._item_indexer.get_items(), weights))
        sorted_items = [k for k, v in sorted(dd.items(), key=lambda item: item[1])]
        return(sorted_items)

        
        


# In[22]:


with open('purchased.json', 'rb') as f:
    purchased_json = json.load(f)
new_model = Recommender()


# In[23]:


new_model.train_new_model(purchased_json)


# In[24]:


x = new_model.predict_all_users()


# In[ ]:




