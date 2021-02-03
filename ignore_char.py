# 训练集或数据集中不包含在 文字集 中的 所有文字 及 数量
import os
from keys import alphabetChinese

root = '/home/hxt/dataset/synth_data/train_200w'
tmp_labels = os.path.join(root, 'tmp_labels.txt')
chars_file = ''

alphabet = alphabetChinese
# with open(chars_file, 'r', encoding='utf-8') as f:
#     while True:
#         line = f.readline()
#         if not line:
#             break
#         alphabet += line[0]
# alphabet = set(alphabet)

ignore_char = set()
count = 0
with open(tmp_labels, 'r', encoding='utf-8') as file:
    for c in file.readlines():
        img_index = c.split(' ', 1)[0]
        if not os.path.exists(os.path.join(root, img_index + '.jpg')):
            continue      
        label = c.split(' ', 1)[-1].rsplit('\n', 1)[0]
        ignore = {l for l in label if l not in alphabet}
        count += 1 if ignore else 0
        ignore_char.update(ignore)
print(ignore_char, count)