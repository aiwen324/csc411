import os
import numpy as np


def read_news(f):
	word_dict = dict()
	for line in f:
		lst_words = line.split()
		for word in lst_words:
			if word in word_dict.keys():
				word_dict[word] += 1
			else:
				word_dict[word] = 1
	return word_dict


# ================================= Part 1 ====================================
real_news_f = open('clean_real.txt', 'r')
fake_news_f = open('clean_fake.txt', 'r')
real_word_dict = read_news(real_news_f)
fake_word_dict = read_news(fake_news_f)
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
        
# =============================== Part 2 ======================================
# Comment: Use numpy permutation to split the training set