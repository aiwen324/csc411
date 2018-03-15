import os

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
for key in l1:
	if key not in l2:
		print key
print "========================================"
for key in l2:
	if key not in l1:
		print key
"""
trump
donald
to
us
trumps
in
on
of
says
for
the
and
with
a
election
clinton
north
as
korea
is
ban
at
will
president
russia
be
over
turnbull
travel
after
deal
white
by
wall
house
from
first
what
not
he
new
china
has
climate
obama
australia
about
how
meeting
call
"""
"""
trump
the
to
in
donald
of
for
a
and
on
is
hillary
clinton
with
will
by
he
election
just
as
new
president
you
it
obama
his
if
at
america
are
be
that
win
from
victory
has
says
what
about
supporters
campaign
news
not
vote
us
after
world
why
i
anti
"""