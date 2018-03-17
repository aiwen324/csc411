import os
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
from pylab import *


def read_news(f):
    news_counter = 0
    word_dict = dict()
    for line in f:
        news_counter += 1
        lst_words = line.split()
        for word in lst_words:
            if word in word_dict.keys():
                word_dict[word] += 1
            else:
                word_dict[word] = 1
    return word_dict, news_counter

def generate_dataset(f, word_list):
    f.seek(0)
    batch_xs = np.empty((0, len(word_list)))
    for line in f:
        news_word = line.split()
        xs = np.zeros(len(word_list))
        for word in news_word:
            xs[word_list.index(word)] = 1
        batch_xs = np.vstack((batch_xs, xs))
    # if real_flag == 1:
    #     batch_ys = np.ones(news_number, dtype=int)
    # else:
    #     batch_ys = np.zeros(news_number, dtype=int)
    return batch_xs

def train_set_word_dict(f, line_arr):
    f.seek(0)
    news_counter = 0
    word_dict = dict()
    for line in f:
        news_counter += 1
        if news_counter in line_arr:
            lst_words = line.split()
            # Just count once for the case that news have duplicate words
            lst_words = set(lst_words)
            for word in lst_words:
                if word in word_dict.keys():
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
    return word_dict, news_counter

# Calculate the conditional probability P(y|X) with Bayes Rule, Here we just
# calculate the enumerator instead of doing the whole things
def naive_bayes(batch_words, prob_log_arr):
    # TODO: Make the function batch_words*prob_log_arr + (1-batch_words)*(1-prob_log_arr)
    # TODO: Take exponetial to the whole array    
    return None

# Calculate the log(p(x_i|y)) and return the whole vector
def calculate_log_probability_array(train_word_dict, train_news_num, word_list, m, p):
    train_keys = train_word_dict.keys()
    prob_arr = np.zeros(len(word_list))
    counter = 0
    for word in word_list:
        if word in train_keys:
            prob_arr[counter] = log(train_word_dict[word] + m*p)-log(train_news_num+m)
        else:
            prob_arr[counter] = log(m*p)-log(train_news_num+m)
    neg_prob_arr = np.ones(len(word_list))
    neg_prob_arr = neg_prob_arr - prob_arr
    return prob_arr, neg_prob_arr
            
def calculate_accuracy():
    return None
        
    
    
# ================================= Part 1 ====================================
real_news_f = open('clean_real.txt', 'r')
fake_news_f = open('clean_fake.txt', 'r')
# Get the word_counter dictionary and amount of news
real_word_dict, real_news_num = read_news(real_news_f)
fake_word_dict, fake_news_num = read_news(fake_news_f)
# Generate word_list, which is a dictionary for word
word_list = real_word_dict.keys()
word_list = word_list + fake_word_dict.keys()
word_list = sorted(word_list)
# Generate dataset for real_news and fake_news
# real_batch_xs = generate_dataset(real_news_f, word_list)
# fake_batch_xs = generate_dataset(fake_news_f, word_list)
# data_set = dict()
# data_set['real_train'] = real_batch_xs[real_news_ind[:int(round(real_news_num*0.7))]]
# data_set['real_valid'] = real_batch_xs[real_news_ind[int(round(real_news_num*0.7)): 
#     int(round(real_news_num*0.85))]]
# data_set['real_test'] = real_batch_xs[real_news_ind[int(round(real_news_num*0.85)):]]

# data_set['fake_train'] = fake_batch_xs[fake_news_ind[:int(round(fake_news_num*0.7))]]
# data_set['fake_valid'] = fake_batch_xs[fake_news_ind[int(round(fake_news_num*0.7)): 
#     int(round(fake_news_num*0.85))]]
# data_set['fake_test'] = fake_batch_xs[fake_news_ind[int(round(fake_news_num*0.85)):]]
# savemat('news_data.mat')
news_number = real_news_num + fake_news_num
real_news_f.close()
fake_news_f.close()
sort_real_keys = sorted(real_word_dict, key=real_word_dict.get, reverse=True)
sort_fake_keys = sorted(fake_word_dict, key=fake_word_dict.get, reverse=True)

real_total = 0
fake_total = 0
for val in real_word_dict.values():
    real_total += val
for val in fake_word_dict.values():
    fake_total += val
    
l1 = sort_real_keys[:50]
l2 = sort_fake_keys[:50]
print "Top 50 word in real news that are not top 50 in fake news"
for key in l1:
    if key not in l2:
        print key, np.true_divide(real_word_dict[key], real_total)
print "========================================"
print "Top 50 word in fake news that are not top 50 in fake news"
for key in l2:
    if key not in l1:
        print key, np.true_divide(fake_word_dict[key], fake_total)

for key in sort_real_keys[:100]:
    if key not in fake_word_dict.keys():
        print key

for key in sort_fake_keys[:100]:
    if key not in real_word_dict.keys():
        print key
        
# =============================== Part 2 ======================================
# Comment: Use numpy permutation to split the training set
# Shuffle the index
data_set = loadmat('news_data.mat')
np.random.seed(0)
real_news_ind = np.random.permutation(real_news_num)
fake_news_ind = np.random.permutation(fake_news_num)
total_word = len(real_word_dict.keys())+len(fake_word_dict.keys())
# Get the train 
real_news_f = open('clean_real.txt', 'r')
fake_news_f = open('clean_fake.txt', 'r')
train_real_news_num = int(round(real_news_num*0.7))
train_fake_news_num = int(round(fake_news_num*0.7))
train_real_word_dict, dummy1 = train_set_word_dict(real_news_f, real_news_ind[:train_real_news_num])
train_fake_word_dict, dummy2 = train_set_word_dict(fake_news_f, fake_news_ind[:train_fake_news_num])
# TODO: Calculate the train dictionary
# TODO: Vectorized the Probability Vector, Remember to take log function
# TODO: Calculate the probability
real_log_prob, neg_real_log_prob = calculate_log_probability_array(train_real_word_dict, 
                                                train_real_news_num, word_list, 
                                                len(word_list), 
                                                np.true_divide(1, len(word_list)))
fake_log_prob, neg_fake_log_prob = calculate_log_probability_array(train_fake_word_dict, 
                                                train_fake_news_num, word_list, 
                                                len(word_list), 
                                                np.true_divide(1, len(word_list)))
real_valid = data_set['real_valid']
fake_valid = data_set['fake_valid']
target_array = np.vstack((np.ones(real_valid.shape[0]), np.zeros(fake_valid.shape[0])))

