import os
import re

f = open('train_ref.txt')
new_f = open('train_new.txt', 'w')

cls_dict = {}
for l in f:
    if l.strip() == "":
        continue
    a, b = l.split(' ')
    match = re.match(r'dataset.bounding_box_train.(\d*).*', a)
    cls_dict[match.group(1)] = b.strip()
    print >>new_f, l.strip()

f.close()

for name in os.listdir('/media/gin/data/re-id'):
    match = re.match(r'(\d*).*', name)
    print >>new_f, "%s %s" % ('dataset/bounding_box_train/' + name, cls_dict[match.group(1)])


