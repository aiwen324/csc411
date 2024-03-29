import os
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
from pylab import *
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from torch.autograd import Variable
import torch
import torchvision
import torch.nn as nn
from sklearn import tree
import graphviz
import commands
import re

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

def Get_stopword_index(word_list):
    counter = 0
    stopword_lst = []
    for word in word_list:
        if word in ENGLISH_STOP_WORDS:
            stopword_lst.append(counter)
        counter += 1
    return stopword_lst
        
    
    
# ================================= Part 1 ====================================
if not os.path.exists('report') or not os.path.isdir('report'):
   os.mkdir('report')
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
# Performance on train set
real_train = data_set['real_train']
fake_train = data_set['fake_train']
train_target_array = np.hstack((np.ones(real_train.shape[0], dtype=int), np.zeros(fake_train.shape[0], dtype=int)))
batch_train = np.vstack((real_train, fake_train))
predict_real_train_bayes = naive_bayes(batch_train, real_log_prob, neg_real_log_prob, train_real_prob)
predict_fake_train_bayes = naive_bayes(batch_train, fake_log_prob, neg_fake_log_prob, train_fake_prob)
combined_predict_train_bayes = np.vstack((predict_fake_train_bayes, predict_real_train_bayes))
predict_train_array = np.argmax(combined_predict_train_bayes, axis=0)
print "Accuracy on train set: ", np.mean(predict_train_array == train_target_array)

# =============================== Part 3a =====================================
# real_log_prob = log(P(x_i|y=real))
# neg_real_log_prob = log(1-P(x_i|y=real)) = log(P(not x_i|y=real))
# fake_log_prob = log(P(x_i|y=fake))
# neg_fake_log_prob = log(1-P(x_i|y=fake)) = log(P(not x_i|y=fake))
def bayes_rule_calculator(cond_prob1, cond_prob2, prob1, prob2):
    cond_prob1 = exp(cond_prob1)
    cond_prob2 = exp(cond_prob2)
    prob = np.true_divide(np.multiply(cond_prob1,prob1), np.multiply(cond_prob1,prob1)+np.multiply(cond_prob2,prob2))
    return prob

p_real_given_word = bayes_rule_calculator(real_log_prob, fake_log_prob, train_real_prob, train_fake_prob)
p_real_given_notword = bayes_rule_calculator(neg_real_log_prob, neg_fake_log_prob, train_real_prob, train_fake_prob)
p_fake_given_word = bayes_rule_calculator(fake_log_prob, real_log_prob, train_fake_prob, train_real_prob)
p_fake_given_notword = bayes_rule_calculator(neg_fake_log_prob, neg_real_log_prob, train_fake_prob, train_real_prob)
# TODO: Print the accuracy
highest_real_index = np.argsort(p_real_given_word)[-10:]
lowest_real_index = np.argsort(p_real_given_notword)[-10:]
highest_fake_index = np.argsort(p_fake_given_word)[-10:]
lowest_fake_index = np.argsort(p_fake_given_notword)[-10:]
# highest_real_index = np.argpartition(p_real_given_word, -10)[-10:]
# lowest_real_index = np.argpartition(neg_real_log_prob, -10)[-10:]
# highest_fake_index = np.argpartition(fake_log_prob, -10)[-10:]
# lowest_fake_index = np.argpartition(neg_fake_log_prob, -10)[-10:]

highest_real = [word_list[i] for i in highest_real_index]
lowest_real = [word_list[i] for i in lowest_real_index]
highest_fake = [word_list[i] for i in highest_fake_index]
lowest_fake = [word_list[i] for i in lowest_fake_index]
highest_real_1 = [(word_list[i], 'probability: '+str(p_real_given_word[i])) for i in highest_real_index]
lowest_real_1 = [(word_list[i], 'probability: '+str(p_real_given_notword[i])) for i in lowest_real_index]
highest_fake_1 = [(word_list[i], 'probability: '+str(p_fake_given_word[i])) for i in highest_fake_index]
lowest_fake_1 = [(word_list[i], 'probability: '+str(p_fake_given_notword[i])) for i in lowest_fake_index]
print "presence most strongly predict real words: ", highest_real_1, '\n'
print "absence most strongly predict real words: ", lowest_real_1, '\n'
print "presence most strongly predict fake words: ", highest_fake_1, '\n'
print "absence most strongly predict fake words: ", lowest_fake_1

# =============================== Part 3b =====================================
# Get stopword_lst
stopword_lst = Get_stopword_index(word_list)
# Copy a numpy array from original
modified_p_real_given_word = np.copy(p_real_given_word)
modified_p_real_given_notword = np.copy(p_real_given_notword)
modified_p_fake_given_word = np.copy(p_fake_given_word)
modified_p_fake_given_notword = np.copy(p_fake_given_notword)
# Modify the stopword value to -infinity
modified_p_real_given_word[stopword_lst] = float('-inf')
modified_p_real_given_notword[stopword_lst] = float('-inf')
modified_p_fake_given_word[stopword_lst] = float('-inf')
modified_p_fake_given_notword[stopword_lst] = float('-inf')
# Get the index again
modified_highest_real_index = np.argsort(modified_p_real_given_word)[-10:]
modified_lowest_real_index = np.argsort(modified_p_real_given_notword)[-10:]
modified_highest_fake_index = np.argsort(modified_p_fake_given_word)[-10:]
modified_lowest_fake_index = np.argsort(modified_p_fake_given_notword)[-10:]

modified_highest_real = [word_list[i] for i in modified_highest_real_index]
modified_lowest_real = [word_list[i] for i in modified_lowest_real_index]
modified_highest_fake = [word_list[i] for i in modified_highest_fake_index]
modified_lowest_fake = [word_list[i] for i in modified_lowest_fake_index]
# The object to print
modified_highest_real_1 = [(word_list[i], 'probability: '+str(modified_p_real_given_word[i])) for i in modified_highest_real_index]
modified_lowest_real_1 = [(word_list[i], 'probability: '+str(modified_p_real_given_notword[i])) for i in modified_lowest_real_index]
modified_highest_fake_1 = [(word_list[i], 'probability: '+str(modified_p_fake_given_word[i])) for i in modified_highest_fake_index]
modified_lowest_fake_1 = [(word_list[i], 'probability: '+str(modified_p_fake_given_notword[i])) for i in modified_lowest_fake_index]
print "======================================================================="
print "presence most strongly predict real words: ", modified_highest_real_1, '\n'
print "absence most strongly predict real words: ", modified_lowest_real_1, '\n'
print "presence most strongly predict fake words: ", modified_highest_fake_1, '\n'
print "absence most strongly predict fake words: ", modified_lowest_fake_1

# =============================== Part 4 ======================================
# We will use Pytorch to do this part

""" Function that calculate the accuracy """
def calculate_accuracy(y_pred, y_target):
    neg_y_pred = np.ones(y_pred.shape[0]) - y_pred
    # print neg_y_pred
    pred_stack = np.vstack((neg_y_pred, y_pred))
    # print pred_stack
    pred = np.argmax(pred_stack, axis=0)
    # print pred
    return np.mean(pred == y_target)
# Loading train set from data_set dictionary
# train
real_train_x = data_set['real_train']
real_train_y = np.ones(real_train_x.shape[0], dtype=int)
fake_train_x = data_set['fake_train']
fake_train_y = np.zeros(fake_train_x.shape[0], dtype=int)
# valid
real_valid_x = data_set['real_valid']
real_valid_y = np.ones(real_valid_x.shape[0], dtype=int)
fake_valid_x = data_set['fake_valid']
fake_valid_y = np.zeros(fake_valid_x.shape[0], dtype=int)
# test
real_test_x = data_set['real_test']
real_test_y = np.ones(real_test_x.shape[0], dtype=int)
fake_test_x = data_set['fake_test']
fake_test_y = np.zeros(fake_test_x.shape[0], dtype=int)

# Combine real and fake train set
train_x = np.vstack((real_train_x, fake_train_x))
valid_x = np.vstack((real_valid_x, fake_valid_x))
test_x = np.vstack((real_test_x, fake_test_x))
train_y = np.hstack((real_train_y, fake_train_y))
valid_y = np.hstack((real_valid_y, fake_valid_y))
test_y = np.hstack((real_test_y, fake_test_y))

# Set the dimension
dim_x = real_train_x.shape[1]
dim_out = 1

# Transform train set to pytorch tensors
dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

train_x_tensor = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
train_y_tensor = Variable(torch.from_numpy(train_y), requires_grad=False).type(dtype_float)

valid_x_tensor = Variable(torch.from_numpy(valid_x), requires_grad=False).type(dtype_float)
# valid_y_tensor = Variable(torch.from_numpy(valid_y), requires_grad=False).type(dtype_float)

test_x_tensor = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
# test_y_tensor = Variable(torch.from_numpy(test_y), requires_grad=False).type(dtype_float)

# Set up Pytorch model
iteration_times = 5000
torch.manual_seed(0)
model = torch.nn.Sequential(
torch.nn.Linear(dim_x, dim_out),
torch.nn.Sigmoid()
)
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
#optimizer = torch.optim.SGD(model.parameters(), lr=5e-5, weight_decay=1e-3)
# TODO: Use a numpy array or dictionary to save the performance of valid and training set
summary = np.empty((0, 2), dtype=float)
for t in range(iteration_times):
    train_y_pred = model(train_x_tensor)
    loss = loss_fn(train_y_pred, train_y_tensor)

    # TODO: Use a numpy array to record the performance
    if t%50 == 0 or t == iteration_times - 1:
    # TODO: convert valid_pred and test_pred to 0 1 array
    # TODO: print some message during the gradient descent
        train_pred_numpy = train_y_pred.data.numpy().flatten()
        valid_pred = model(valid_x_tensor).data.numpy().flatten()
        test_pred = model(test_x_tensor).data.numpy().flatten()

        train_perform = calculate_accuracy(train_pred_numpy, train_y)
        valid_perform = calculate_accuracy(valid_pred, valid_y)
        test_perform = calculate_accuracy(test_pred, test_y)

        summary = np.vstack((summary, np.array([valid_perform, test_perform])))

        print "Iteration: " + str(t)
        print "loss is: " + str(loss.data[0])
        print "train_perform: " + str(train_perform)
        print "valid_perform: " + str(valid_perform)
        if t == iteration_times - 1:
            print "final test performance: " + str(test_perform)

    model.zero_grad()
    loss.backward()
    optimizer.step()

xs = np.array([50*i for i in range(summary.shape[0])])
valid_perform_ys = summary[:, 0]
test_perform_ys = summary[:, 1]
plt.plot(xs, test_perform_ys, 'g', label='test')
plt.plot(xs, valid_perform_ys, 'r', label='validate')
plt.xlabel('iteration')
plt.ylabel('Performance')
plt.title("Performance on lr = " + '1e-4')
plt.legend()
plt.savefig('report/part4.png')
plt.show()

# =============================== Part 6a =====================================
net_weight = model[0].weight.data.numpy()
net_weight = net_weight.flatten()
highest_thetas_index = np.argsort(net_weight)[-10:]
highest_thetas = [word_list[i] for i in highest_thetas_index]
print "Highest 10 thetas index corresponding word: "
for i in highest_thetas_index:
    print 'Word: ', word_list[i], '\t theta value: ', net_weight[i] 
lowest_thetas_index = np.argsort(net_weight)[:10]
lowest_thetas = [word_list[i] for i in lowest_thetas_index]
print "Lowest 10 thetas index corresponding word: "
for i in lowest_thetas_index:
    print 'Word: ', word_list[i], '\t theta value: ', net_weight[i]

highest_overlap = set(highest_thetas) & (set(highest_real) | set(lowest_fake))
lowest_overlap = set(lowest_thetas) & (set(lowest_real) | set(highest_fake))

print "Overlap {} words for highest: {}".format(len(highest_overlap), str(highest_overlap))
print "Overlap {} words for lowest: {}".format(len(lowest_overlap), str(lowest_overlap))
# =============================== Part 6b =====================================
stopword_lst = Get_stopword_index(word_list)
net_weight_copy1 = np.copy(net_weight)
net_weight_copy1[stopword_lst] = float('-inf')
highest_thetas_nostopwords_index = np.argsort(net_weight_copy1)[-10:]
highest_thetas_nostopwords = [word_list[i] for i in highest_thetas_nostopwords_index]
print "Highest 10 thetas index corresponding word: "
for i in highest_thetas_nostopwords_index:
    print 'Word: ', word_list[i], '\t theta value: ', net_weight[i]

net_weight_copy2 = np.copy(net_weight)
net_weight_copy2[stopword_lst] = float('inf')
lowest_thetas_nostopwords_index = np.argsort(net_weight_copy2)[:10]
lowest_thetas_nostopwords = [word_list[i] for i in lowest_thetas_nostopwords_index]
print "Lowest 10 thetas index corresponding word: "
for i in lowest_thetas_nostopwords_index:
    print 'Word: ', word_list[i], '\t theta value: ', net_weight[i] 

highest_overlap = set(highest_thetas_nostopwords) & (set(modified_highest_real) | set(modified_lowest_fake))
lowest_overlap = set(lowest_thetas_nostopwords) & (set(modified_lowest_real) | set(modified_highest_fake))

print "Overlap {} words for highest: {}".format(len(highest_overlap), str(highest_overlap))
print "Overlap {} words for lowest: {}".format(len(lowest_overlap), str(lowest_overlap))

# =============================== Part 7a =====================================
# train_x, valid_x and test_x is from part 4
i = 33
clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best',
                                  max_depth=i, min_samples_split=2,
                                  min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.0, max_features=317, 
                                  random_state=0, max_leaf_nodes=None, 
                                  min_impurity_decrease=0.0, min_impurity_split=None,
                                  class_weight=None, presort=False)
clf = clf.fit(train_x, train_y)
print "max_dept: {}, max_features: {}".format(i, 317)
predict_train_y = clf.predict(train_x)
print "Performance on Train is: ", np.mean(predict_train_y == train_y)
predict_valid_y = clf.predict(valid_x)
print "Performance on Valid is: ", np.mean(predict_valid_y == valid_y)
predict_test_y = clf.predict(test_x)
print "Performance on Test is: ", np.mean(predict_test_y == test_y)
print "========================= "
filename = 'test_complete'
dot_data = tree.export_graphviz(clf, out_file='report/'+ filename +'.dot',
                                max_depth=None, feature_names=word_list, 
                                class_names=['fake', 'real'])
# Just make it easy to copy the command to shell
print 'dot -Tpng '+filename+'.dot'+' -o '+filename+'.png'

# =============================== Part 7b =====================================
tree_file = open('report/test_complete.dot')
count = 0
top_words = []
for line in tree_file:
    if 'label' in line:
        m1 = re.search('label="', line)
        m2 = re.search(r' <= 0.5\\ngini', line)
        if m2 is not None:
            w = line[m1.end(): m2.start()]
            top_words.append(w)
        count += 1
    if count > 200:
        break
tree_file.close()
overlap_with_thetas = set(top_words) & \
                      (set(highest_thetas) | 
                       set(lowest_thetas))
print "Top words in decision tree and thetas words has the following overlapping:"
print len(overlap_with_thetas), "words are overlapped, they are: "
print overlap_with_thetas

overlap_with_bayes = set(top_words) & \
                    (set(highest_real)|set(lowest_real)|set(highest_fake)|
                            set(lowest_fake))
print "Top words in decision tree and top bayes words has the following overlapping:"
print len(overlap_with_thetas), "words are overlapped, they are: "
print overlap_with_bayes

# =============================== Part 8 ======================================
def calculate_mutual_information(top, left, right):
    P_Y = np.true_divide(top[1], top[0]+top[1])
    P_not_Y = 1 - P_Y
    # P(x_i=0)
    P_left = np.true_divide(left[0]+left[1], left[0]+left[1]+right[0]+right[1])
    # P(x_i=1)
    P_right = 1 - P_left
    # P(Y=real|x_i=0)
    P_Y_gv_left = np.true_divide(left[1], left[0]+left[1])
    # P(Y=fake|x_i=0)
    P_not_Y_gv_left = 1 - P_Y_gv_left
    # P(Y=real|x_i=1)
    P_Y_gv_right = np.true_divide(right[1], right[0]+right[1])
    # P(Y=fake|x_i=1)
    P_not_Y_gv_right = 1 - P_Y_gv_right
    # H(Y|x_i=0)
    if P_Y_gv_left == 1 or P_Y_gv_left == 0:
        H_Y_gv_left = 0
    else:
        H_Y_gv_left = -(P_Y_gv_left*np.log2(P_Y_gv_left)+P_not_Y_gv_left*np.log2(P_not_Y_gv_left))
    # H(Y|x_i=1)
    if P_Y_gv_right == 1 or P_Y_gv_right == 0:
        H_Y_gv_right = 0
    else:
        H_Y_gv_right = -(P_Y_gv_right*np.log2(P_Y_gv_right)+P_not_Y_gv_right*np.log2(P_not_Y_gv_right))
    # H(Y)
    if P_Y == 0 or P_Y == 1:
        H_Y = 0
    else:
        H_Y = -(P_Y*np.log2(P_Y)+P_not_Y*np.log2(P_not_Y))
    # H(Y|x_i)
    H_Y_gv_X = P_left*H_Y_gv_left + P_right*H_Y_gv_right
    return H_Y - H_Y_gv_X
