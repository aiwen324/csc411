import os
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
from pylab import *
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from torch.autograd import Variable
import torch
import torchvision
import torch.nn as nn

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
# calculate the enumerator instead of doing the whole fraction calculation
def naive_bayes(batch_words, log_prob, neg_log_prob, p_y):
    neg_batch_words = np.ones(batch_words.shape)
    neg_batch_words = neg_batch_words - batch_words
    # Make the function batch_words*log_prob + (1-batch_words)*(1-log_prob)
    predict = np.dot(batch_words, log_prob) + np.dot(neg_batch_words, neg_log_prob)
    # Take exponetial to the whole array    
    predict = exp(predict)
    return predict*p_y

# Calculate the log(p(x_i|y)) and return the whole vector
def calculate_log_probability_array(train_word_dict, train_news_num, word_list, m, p_hat):
    train_keys = train_word_dict.keys()
    prob_arr = np.zeros(len(word_list))
    neg_prob_arr = np.zeros(len(word_list))
    counter = 0
    for word in word_list:
        if word in train_keys:
            prob_arr[counter] = log(train_word_dict[word] + m*p_hat)-log(train_news_num+m)
            neg_prob_arr[counter] = log(1-np.true_divide(train_word_dict[word] + m*p_hat, train_news_num+m))
        else:
            prob_arr[counter] = log(m*p_hat)-log(train_news_num+m)
            neg_prob_arr[counter] = log(1-np.true_divide(m*p_hat, train_news_num+m))
        counter += 1
    return prob_arr, neg_prob_arr
            
def calculate_accuracy():
    return None

def Get_stopword_index(word_list):
    counter = 0
    stopword_lst = []
    for word in word_list:
        if word in ENGLISH_STOP_WORDS:
            stopword_lst.append(counter)
        counter += 1
    return stopword_lst
        
    
    
# ================================= Part 1 ====================================
real_news_f = open('clean_real.txt', 'r')
fake_news_f = open('clean_fake.txt', 'r')
# Get the word_counter dictionary and amount of news
real_word_dict, real_news_num = read_news(real_news_f)
fake_word_dict, fake_news_num = read_news(fake_news_f)
# Generate word_list, which is a dictionary for word
word_list = set(real_word_dict.keys())
word_list = word_list.union(set(fake_word_dict.keys()))
word_list = sorted(list(word_list))
# Shuffle the index
np.random.seed(0)
real_news_ind = np.random.permutation(real_news_num)
fake_news_ind = np.random.permutation(fake_news_num)
# # Generate dataset for real_news and fake_news
# real_batch_xs = generate_dataset(real_news_f, word_list)
# fake_batch_xs = generate_dataset(fake_news_f, word_list)
# # Save to the dictionary
# data_set = dict()
# data_set['real_train'] = real_batch_xs[real_news_ind[:int(round(real_news_num*0.7))]]
# data_set['real_valid'] = real_batch_xs[real_news_ind[int(round(real_news_num*0.7)): 
#     int(round(real_news_num*0.85))]]
# data_set['real_test'] = real_batch_xs[real_news_ind[int(round(real_news_num*0.85)):]]

# data_set['fake_train'] = fake_batch_xs[fake_news_ind[:int(round(fake_news_num*0.7))]]
# data_set['fake_valid'] = fake_batch_xs[fake_news_ind[int(round(fake_news_num*0.7)): 
#     int(round(fake_news_num*0.85))]]
# data_set['fake_test'] = fake_batch_xs[fake_news_ind[int(round(fake_news_num*0.85)):]]
# savemat('news_data.mat', data_set)
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
data_set = loadmat('news_data.mat')
# Get the train 
real_news_f = open('clean_real.txt', 'r')
fake_news_f = open('clean_fake.txt', 'r')
train_real_news_num = int(round(real_news_num*0.7))
train_fake_news_num = int(round(fake_news_num*0.7))
train_real_word_dict, dummy1 = train_set_word_dict(real_news_f, real_news_ind[:train_real_news_num])
train_fake_word_dict, dummy2 = train_set_word_dict(fake_news_f, fake_news_ind[:train_fake_news_num])
# Extract input from dictionary and do some basic caluculation
real_valid = data_set['real_valid']
fake_valid = data_set['fake_valid']
valid_target_array = np.hstack((np.ones(real_valid.shape[0], dtype=int), np.zeros(fake_valid.shape[0], dtype=int)))
batch_valid = np.vstack((real_valid, fake_valid))
train_real_prob = np.true_divide(train_real_news_num, train_real_news_num+train_fake_news_num)
train_fake_prob = 1 - train_real_prob
# Tune the m and p_hat here
real_log_prob, neg_real_log_prob = calculate_log_probability_array(train_real_word_dict, 
                                                train_real_news_num, word_list, 
                                                len(word_list)*0.01, 
                                                np.true_divide(1, len(word_list)))
fake_log_prob, neg_fake_log_prob = calculate_log_probability_array(train_fake_word_dict, 
                                                train_fake_news_num, word_list, 
                                                len(word_list)*0.01, 
                                                np.true_divide(1, len(word_list)))
predict_real_valid_bayes = naive_bayes(batch_valid, real_log_prob, neg_real_log_prob, train_real_prob)
predict_fake_valid_bayes = naive_bayes(batch_valid, fake_log_prob, neg_fake_log_prob, train_fake_prob)
combined_predict_valid_bayes = np.vstack((predict_fake_valid_bayes, predict_real_valid_bayes))
predict_valid_array = np.argmax(combined_predict_valid_bayes, axis=0)
print "Accuracy on valid set: ", np.mean(predict_valid_array == valid_target_array)
# After tuning the m and p_hat
real_test = data_set['real_test']
fake_test = data_set['fake_test']
test_target_array = np.hstack((np.ones(real_test.shape[0], dtype=int), np.zeros(fake_test.shape[0], dtype=int)))
batch_test = np.vstack((real_test, fake_test))
predict_real_test_bayes = naive_bayes(batch_test, real_log_prob, neg_real_log_prob, train_real_prob)
predict_fake_test_bayes = naive_bayes(batch_test, fake_log_prob, neg_fake_log_prob, train_fake_prob)
combined_predict_test_bayes = np.vstack((predict_fake_test_bayes, predict_real_test_bayes))
predict_test_array = np.argmax(combined_predict_test_bayes, axis=0)
print "Accuracy on test set: ", np.mean(predict_test_array == test_target_array)

# =============================== Part 3a =====================================
highest_real_index = np.argpartition(real_log_prob, -10)[-10:]
lowest_real_index = np.argpartition(neg_real_log_prob, -10)[-10:]
highest_fake_index = np.argpartition(fake_log_prob, -10)[-10:]
lowest_fake_index = np.argpartition(neg_fake_log_prob, -10)[-10:]
print "presence most strongly predict real words: ", [word_list[i] for i in highest_real_index]
print "absence most strongly predict real words: ", [word_list[i] for i in lowest_real_index]
print "presence most strongly predict fake words: ", [word_list[i] for i in highest_fake_index]
print "absence most strongly predict fake words: ", [word_list[i] for i in lowest_fake_index]

# =============================== Part 3b =====================================
# Get stopword_lst
stopword_lst = Get_stopword_index(word_list)
# Copy a numpy array from original
modified_real_log_prob = np.copy(real_log_prob)
modified_neg_real_log_prob = np.copy(neg_real_log_prob)
modified_fake_log_prob = np.copy(fake_log_prob)
modified_neg_fake_log_prob = np.copy(neg_fake_log_prob)
# Modify the stopword value to -infinity
modified_real_log_prob[stopword_lst] = float('-inf')
modified_neg_real_log_prob[stopword_lst] = float('-inf')
modified_fake_log_prob[stopword_lst] = float('-inf')
modified_neg_fake_log_prob[stopword_lst] = float('-inf')
# Get the index again
modified_highest_real_index = np.argpartition(modified_real_log_prob, -10)[-10:]
modified_lowest_real_index = np.argpartition(modified_neg_real_log_prob, -10)[-10:]
modified_highest_fake_index = np.argpartition(modified_fake_log_prob, -10)[-10:]
modified_lowest_fake_index = np.argpartition(modified_neg_fake_log_prob, -10)[-10:]
print "presence most strongly predict real words: ", [word_list[i] for i in modified_highest_real_index]
print "absence most strongly predict real words: ", [word_list[i] for i in modified_lowest_real_index]
print "presence most strongly predict fake words: ", [word_list[i] for i in modified_highest_fake_index]
print "absence most strongly predict fake words: ", [word_list[i] for i in modified_lowest_fake_index]

# =============================== Part 4 ======================================
dim_x = len(word_list)
dim_out = 1
torch.manual_seed(0)
model = torch.nn.Sequential(
torch.nn.Linear(dim_x, dim_out),
torch.nn.Sigmoid()
)
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
x = Variable(torch.from_numpy())