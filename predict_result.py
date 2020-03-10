import codecs

f1 = codecs.open(r'.\checkpoint\label_test1.txt', 'r', encoding='utf8')
f2 = codecs.open(r'.\data\test.txt', 'r', encoding='utf8')
f3 = codecs.open(r'.\checkpoint\result_test.txt', 'w', encoding='utf8')
predect = []
tmp = []
for label in f1:
    label = label.strip()
    if label == '[CLS]':
        tmp = []

    elif label == '[SEP]':
        predect.append(' '.join(tmp))

    else:
        tmp.append(label)

words = []
labels = []
for line in f2:
    word = line.strip().split('-seq-')[0]
    words.append(word)

    label = line.strip().split('-seq-')[1]
    labels.append(label)

assert len(words) == len(predect)

for i in range(len(labels)):
    print('输入：', words[i], len(words[i].split(' ')))
    print('标签：', labels[i], len(labels[i].split(' ')))
    print('预测：', predect[i], len(predect[i].split(' ')))
    print('-'*10)

