# https://arxiv.org/abs/1508.07909 Byte-Pair Encoding (BPE)

import re, collections
import numpy as np # 1.15
import csv
import time
import os
from tqdm import tqdm


def save_voca(path, data):
	with open(path, 'w', encoding='utf-8') as o:
		for i in data:
			line = ''
			for j in i: 
				line += str(j) + ' '
			o.write(line+'\n')


def read_voca(path):
	sorted_voca = []
	with open(path, 'r', encoding='utf-8') as f:	
		for bpe_voca in f:
			bpe_voca = bpe_voca.strip()
			if bpe_voca:
				bpe_voca = bpe_voca.split()
				sorted_voca.append(bpe_voca)
	return sorted_voca



def merge(word, sorted_voca, space_symbol='</w>'):
	#word: 'test'
	# 't est</w>', 'te st</w>', 'tes t</w>', 'test </w>'

	# word+space_symbol이 high_freq_voca에 있으면 리턴.
	# 없으면 word+space_symbol을 prefix, suffix로 나누고,  prefix, suffix 모두 high_freq_voca에 있으면 리턴
	# suffix가 high_freq_voca에는 없는경우, prefix가 high_freq_voca에 있거나 len(prefix)가 1이면, 그 때의 prefix, suffix 저장.
		# ==> 최종적으로 prefix 길이가 가장 길면서 위 조건 만족하는게 저장됨.(규칙이 정해짐: prefix가 가장 길고 voco에 있는 경우, 즉 딥러닝이 예측할땐 prefix가 가장 긴것을 예측할 것으로 기대.)
		# ==> 이제 suffix만 처리하면 되는데 suffix를 word로 해서 재귀적으로 같은 방식으로 처리함.
	if word+space_symbol in sorted_voca:
		return word+space_symbol

	for i in range(1, len(word)+1):
		prefix = word[:i]
		suffix = word[i:]+space_symbol

		if prefix in sorted_voca and suffix in sorted_voca:
			return prefix + ' ' + suffix

		if prefix in sorted_voca or len(prefix)==1:
			prefix_result = prefix
			suffix_result = suffix

	if suffix_result == space_symbol:
		return prefix + ' ' + suffix_result

	return prefix_result + ' ' + merge(suffix_result[:-len(space_symbol)], sorted_voca)



def get_vocabulary(path_list):
	# get voca freq
	word_frequency_dict = {}
	for path in path_list:
		with open(path, 'r', encoding='utf-8') as f:
			documents = f.readlines()

		for i in tqdm(range(len(documents)), ncols=50):
			sentence = documents[i]
			for word in sentence.strip().split():
				if word in word_frequency_dict:
					word_frequency_dict[word] += 1
				else:
					word_frequency_dict[word] = 1

	sorted_voca = sorted(tuple(word_frequency_dict.items()), key=lambda x: x[1], reverse=True)
	return sorted_voca


def _apply_bpe(path, out_path, space_symbol='</w>', sorted_voca={}):
	# write file
	o = open(out_path, 'w', newline='', encoding='utf-8')
	wr = csv.writer(o, delimiter=' ')

	with open(path, 'r', encoding='utf-8') as f:
		documents = f.readlines()

	for i in tqdm(range(len(documents)), ncols=50):
		row = []
		sentence = documents[i]

		for word in sentence.strip().split():
			bpe = merge(word, sorted_voca, space_symbol)
			row.extend(bpe.split())
		wr.writerow(row)

	o.close()


def apply_bpe(path_list, out_list, voca_path, new_voca_path=None, final_voca_threshold=1, final_voca_num=None, space_symbol='</w>'):
	# final_voca_threshold: final voca에 참여시킬 voca의 threshold
	print('apply bpe')


	sorted_voca = read_voca(voca_path)
	print('loaded_voca size:', len(sorted_voca))
	
	if final_voca_num:
		sorted_voca = dict(sorted_voca[:final_voca_num])
	else:		
		if final_voca_threshold > 1:
			sorted_voca = [(word, int(freq)) for (word, freq) in sorted_voca if int(freq) >= final_voca_threshold]
			print('threshold applied voca size:', len(sorted_voca))
		sorted_voca = dict(sorted_voca)


	for i in range(len(path_list)):
		path = path_list[i]
		out_path = out_list[i]

		directory_of_out_path = os.path.split(out_path)[0] # 0: directory, 1: filename
		if not os.path.exists(directory_of_out_path):
			print("create" + directory_of_out_path + "directory")
			os.makedirs(directory_of_out_path)

		print('path:', path, ', out_path:', out_path)
		_apply_bpe(
				path=path, 
				out_path=out_path,
				space_symbol=space_symbol, 
				sorted_voca=sorted_voca
			)


	if new_voca_path:
		bpe_path_list = [path for path in out_list]
		if final_voca_num:
			new_sorted_voca = get_vocabulary(bpe_path_list)[:final_voca_num]
		else:
			new_sorted_voca = get_vocabulary(bpe_path_list)

		save_voca(new_voca_path, new_sorted_voca)
		print(new_voca_path, "data size:", len(new_sorted_voca))

	print('\n\n\n')

