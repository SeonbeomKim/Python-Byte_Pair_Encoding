# https://arxiv.org/abs/1508.07909 Byte-Pair Encoding (BPE)
# https://lovit.github.io/nlp/2018/04/02/wpm/ 참고

import re, collections
import numpy as np # 1.15
import csv
import time
import os
from tqdm import tqdm



def save_data(path, data):
	np.save(path, data)

def load_data(path, mode=None):
	data = np.load(path, encoding='bytes')
	if mode == 'dictionary':
		data = data.item()
	return data

def save_list_to_txt(path, data):
	with open(path, 'w', encoding='utf-8') as o:
		for i in data:
			line = ''
			for j in i: 
				line += str(j) + ' '
			o.write(line+'\n')


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


def apply_bpe(path_list, out_bpe_path, out_list, voca_npy_path, new_voca_npy_path=None, final_voca_threshold=1, space_symbol='</w>'):
	# final_voca_threshold: final voca에 참여시킬 voca의 threshold
	print('apply bpe')

	if not os.path.exists(out_bpe_path):
		print("create" + out_bpe_path + "directory")
		os.makedirs(out_bpe_path)

	sorted_voca = load_data(voca_npy_path)
	print('loaded_voca size:', len(sorted_voca))
	
	if final_voca_threshold > 1:
		sorted_voca = [(word, int(freq)) for (word, freq) in sorted_voca if int(freq) >= final_voca_threshold]
		print('threshold applied voca size:', len(sorted_voca))
	sorted_voca = dict(sorted_voca)

	for i in range(len(path_list)):
		path = path_list[i]
		out_path = out_list[i]

		print('path:', path, ', out_path:', out_path)
		_apply_bpe(
				path=path, 
				out_path=out_bpe_path+out_path,
				space_symbol=space_symbol, 
				sorted_voca=sorted_voca
			)

	if new_voca_npy_path:
		bpe_path_list = [out_bpe_path+path for path in out_list]
		new_sorted_voca = get_vocabulary(bpe_path_list)
		save_data(new_voca_npy_path, new_sorted_voca)
		save_list_to_txt(new_voca_npy_path.replace('.npy', '.txt'), new_sorted_voca)
		print(new_voca_npy_path, "data size:", len(new_sorted_voca))

	print('\n\n\n')



"""
npy_path = './npy/' # path of bpe2idx, idx2bpe, merge_info and cache 
out_bpe_path = './bpe_dataset/' # bpe applied path
voca_npy_path = npy_path+'voca_from_learn_BPE.npy'
voca_npy_path_semi_final = npy_path+'semi_final_voca.npy'
voca_npy_path_final = npy_path+'final_voca.npy'

# train data path
path_list = ["./dataset/corpus.tc.en/corpus.tc.en", "./dataset/corpus.tc.de/corpus.tc.de"] # original data1, data2
out_list = ['./bpe_wmt17.en', './bpe_wmt17.de'] # bpe_applied_data1, data2

# test data path
test_path_list = [
		'./dataset/dev.tar/newstest2014.tc.en',
		'./dataset/dev.tar/newstest2015.tc.en',
		'./dataset/dev.tar/newstest2016.tc.en',
	]
test_out_list = [
		'./bpe_newstest2014.en', 
		'./bpe_newstest2015.en', 
		'./bpe_newstest2016.en', 
	] 		


final_voca_threshold = 50

# bpe 적용하고 모든 bpe 단어 빈도수대로 추출 
	# 기존의 learn_BPE에서 생성된 voca의 freq와 다른 freq의 voca가 생성됨.(apply_BPE의 merge 방식이 learn_BPE의 merge_info 순서대로 하지 않기 때문임.)
apply_bpe(
		path_list=path_list, 
		out_bpe_path=out_bpe_path, 
		out_list=out_list, 
		voca_npy_path=voca_npy_path, 
		new_voca_npy_path=voca_npy_path_semi_final, 
		final_voca_threshold=1,
		space_symbol='</w>'
	)

# 적용된 bpe 단어에서 빈도수대로 끊고 다시 적용 => reapply_bpe
	# apply_BPE 에서 사용된 merge로 부터 생성된 voca중에 freq가 낮은건 버리고, apply bpe 다시 적용. 여기서 생성되는 voca가 Final voca임. 앞으로 모두 이 voca 쓰면 됨.
apply_bpe(
		path_list=path_list, 
		out_bpe_path=out_bpe_path, 
		out_list=out_list, 
		voca_npy_path=voca_npy_path_semi_final, 
		new_voca_npy_path=voca_npy_path_final, 
		final_voca_threshold=final_voca_threshold, 
		space_symbol='</w>'
	)

# testset bpe apply
apply_bpe(
		path_list=test_path_list, 
		out_bpe_path=out_bpe_path, 
		out_list=test_out_list, 
		voca_npy_path=voca_npy_path_final, 
		new_voca_npy_path=None,
		final_voca_threshold=final_voca_threshold, 
		space_symbol='</w>'
	)

"""