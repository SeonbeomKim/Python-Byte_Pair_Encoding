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


# word:"abc" => "a b c space_symbol"
def word_split_for_bpe(word, space_symbol='</w>'):
	return ' '.join(list(word)) + ' ' + space_symbol



# word frequency 추출.
def get_word_frequency_dict_from_document(path, space_symbol='</w>'):
	word_frequency_dict = {}

	with open(path, 'r', encoding='utf-8') as f:
		documents = f.readlines()

	for i in tqdm(range(len(documents)), ncols=50):
		sentence = documents[i]

		for word in sentence.strip().split():
			word = word_split_for_bpe(word, space_symbol) # "abc" => "a b c space_symbol"
			if word in word_frequency_dict:
				word_frequency_dict[word] += 1
			else:
				word_frequency_dict[word] = 1
	return word_frequency_dict




# merge two dictionary
def merge_dictionary(dic_a, dic_b):
	for i in dic_b:
		if i in dic_a:
			dic_a[i] += dic_b[i]
		else:
			dic_a[i] = dic_b[i]
	return dic_a


# 2-gram frequency table 추출.
def get_stats(word_frequency_list):
	pairs = {}
	for word, freq in word_frequency_list:
		symbols = word.split()
		for i in range(len(symbols)-1):
			gram = (symbols[i],symbols[i+1])
			if gram in pairs:
				pairs[gram] += freq
			else:
				pairs[gram] = freq
	return pairs # tuple을 담고 있는 dictionary 리턴.


def delete_some_stats(stats, best_pair):
	# ac t c t s   [c t]   => ac t ct s
	#   stats    best_pair   new_stats
	# [ac, t]	  [c, t]		[ac, t]
	# [t, c]					[t, ct]
	# [c, t]					[ct, s]
	# [t, s]

	# left, right = best_pair 라고 할 때,
	# left == info[1] 이거나 right == info[0] 이거나 
	# (left==info[0] and right==info[1]) 이면 stats에서 제거

	# 만약 best_pair[0]이 stats[1]에 있으면 원래 stats[1] 뒤에 best_pair[1]이 있었으면
	# 저 stats 가 만들어질 수 없으므로 재계산.
	# 또한 best_pair[1]이 stats[0]에 있으면 원래 stats[0] 앞에 best_pair[1]이 있었다면
	# 저 stats가 만들어질 수 없으므로 재계산.
	# 또한 stats가 best_pair랑 동일하면 재계산

	
	left, right = best_pair	
	del stats[best_pair] # (left == info[0] and right == info[1])
	
	for info in list(stats.keys()):
		if left == info[1] or right == info[0]:
			del stats[info]
			#print('delete from', 'info:',info, 'best_pair:', best_pair)
	return stats



# 2-gram frequency table 추출. (불필요한 2gram freq는 재계산 X)
def selective_get_stats(data):
	best_pair = data[0]
	word_frequency_list = data[1] 

	left, right = best_pair	
	best_pair_to_string = left+right

	# word_frequency_list는 best_pair 기준으로 합쳐졌으므로, best_pair_to_string이 포함된것 계산
	# 또한 left == info[1] 이거나 right == info[0] 인것을 제거했으므로 이것들만 계산.
	# 다른경우는 계산 x

	stats = {}
	for word, freq in word_frequency_list:
		symbols = word.split()
		if left in symbols or right in symbols or best_pair_to_string in symbols:
			for i in range(len(symbols)-1):
				gram = (symbols[i],symbols[i+1])
				if left == gram[1] or right == gram[0] or best_pair_to_string in gram:
					if gram in stats:
						stats[gram] += freq
					else:
						stats[gram] = freq
	return stats
	


# pairs 중에서 가장 높은 frequency를 갖는 key 리턴.
def check_merge_info(pairs):
	best = max(pairs, key=pairs.get)
	return best



# frequency가 가장 높은 best_pair 정보를 이용해서 단어를 merge.
def merge_bpe_word(best_pair_and_word_frequency_list):
	best_pair = best_pair_and_word_frequency_list[0] # tuple ('r','</w>')
	word_frequency = best_pair_and_word_frequency_list[1] # list

	v_out = []

	best_pair_to_string_with_space = ' '.join(best_pair)
	best_pair_to_string = ''.join(best_pair) # ['c', 'e'] => 'ce'

	bigram = re.escape(best_pair_to_string_with_space)
	p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
	
	for word, freq in word_frequency:
		#if best_pair_to_string_with_space in word: 
		if ' '+best_pair_to_string_with_space+' ' in ' '+word+' ': 
			w_out = p.sub(best_pair_to_string, word) # 만약 ''.join(best_pair): r</w> 이고, word: 'a r </w>' 이면 w_out은 'a r</w>'가 된다.
			v_out.append( (w_out, freq) )
		else:
			v_out.append( (word, freq) )

	return v_out



def merge_a_word(merge_info, word, cache={}, high_freq_voca={}):
	# merge_info: list, ['c', 'e']
	# word: "c e m e n t </w>" => "ce m e n t<\w>" 되어야 함.
	
	if word in cache:
		return cache[word]

	remove_space = word.replace(' ', '')
	if remove_space in high_freq_voca:
		cache[word] = remove_space
		return remove_space

	else:
		bpe_word = word
		for i, info in enumerate(merge_info):
			# 처음엔 무조건 result 초기화됨.  
			# merge하다가 high_freq_voca에 없는 방향으로 merge가 되면 result에 반영이 안됨.
			# 즉 가장 최선의 merge 결과 저장.
			split_bpe_word = bpe_word.split()
			if all(sub in high_freq_voca or len(sub)==1 or sub=='</w>' for sub in split_bpe_word):
				result = bpe_word

			if len(split_bpe_word) == 1: # 더이상 merge할 것이 없는 상황.
				break

			info_to_string_with_space = ' '.join(info)
			#if info_to_string_with_space in bpe_word: 
			if ' '+info_to_string_with_space+' ' in ' '+bpe_word+' ': 
				bigram = re.escape(info_to_string_with_space)
				p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
				bpe_word = p.sub(''.join(info), bpe_word) # 만약 info_to_string_with_space: 'r </w>' 이고, bpe_word: 'a r </w>' 이면 w_out은 'a r</w>'가 된다.

		# cache upate
		cache[word] = result # 모든 segment가 high_freq_voca에 있거나 분할 불가능인 상황, 즉 최선의 경우만 저장함.
		return result


def _make_total_word_cache_before_apply_bpe(data):
	merge_info = data[0] 
	word_list = data[1]
	high_freq_voca = data[2] 

	# merge_info: list, ['c', 'e']
	# word: "c e m e n t </w>" 형태 
	# high_freq_voca: dict 형태 {voca:freq}

	cache = {}
	word_list_size = len(word_list)
	for i in tqdm(range(word_list_size), ncols=50):
		bpe_word = word_list[i]
		
		remove_space = bpe_word.replace(' ', '')
		if remove_space in high_freq_voca:
			result = remove_space
		
		else:
			for info in merge_info:

				split_bpe_word = bpe_word.split()
				if all(sub in high_freq_voca or len(sub)==1 or sub=='</w>' for sub in split_bpe_word):
					result = bpe_word

				if len(split_bpe_word) == 1: # 더이상 merge할 것이 없는 상황.
					break
			
				info_to_string_with_space = ' '.join(info)
				#if info_to_string_with_space in bpe_word: 
				if ' '+info_to_string_with_space+' ' in ' '+bpe_word+' ': 
					bigram = re.escape(info_to_string_with_space)
					p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
					bpe_word = p.sub(''.join(info), bpe_word)

		cache[word_list[i]] = result
	return cache


def make_total_word_cache_before_apply_bpe(path_list, npy_path, space_symbol='</w>', multi_proc=1):
	print('make_total_word_cache_before_apply_bpe','\n')

	print('get word frequency dictionary')
	total_word_frequency_dict = {}
	for path in path_list:
		word_frequency_dict = get_word_frequency_dict_from_document(
				path=path, 
				space_symbol=space_symbol, 
			) #ok
		total_word_frequency_dict = merge_dictionary(total_word_frequency_dict, word_frequency_dict)

	total_words = list(total_word_frequency_dict.keys())
	print('total_words size:', len(total_words))

	merge_info = load_data(npy_path+'merge_info.npy')
	sorted_voca = load_data(npy_path+'sorted_voca.npy')
	high_freq_voca = [(word, int(freq)) for (word, freq) in sorted_voca if int(freq) >= final_voca_threshold]
	high_freq_voca = dict(sorted_voca)

	if os.path.isfile(npy_path+"cache.npy"):
		cache = load_data(npy_path+'cache.npy', mode='dictionary')
	else:
		cache = {}
	print('current cache_size:', len(cache), '\n')

	# init multiprocessing
	if multi_proc > 1:
		process = multi_proc
		print('# process:', process, '\n')
		pool = mp.Pool(process)

		slice_size =  len(total_words)//process
		slicing = [k*slice_size for k in range(process)]
		slicing.append(len(total_words)) # [0, @@, ..., len(word_frequency_dict)] # 총 process+1개 
		print('multiproc data slicing boundary:', slicing)

		multi_merge_info = [merge_info]*process
		multi_high_freq_voca = [high_freq_voca]*process
		results = pool.map(
				_make_total_word_cache_before_apply_bpe, 
				zip( multi_merge_info, [total_words[slicing[k]:slicing[k+1]] for k in range(process)], multi_high_freq_voca )
			)
		for dic in results:
			cache.update(dic)

		pool.close()

	else:
		cache = _make_total_word_cache_before_apply_bpe([merge_info, total_words, high_freq_voca])

	save_data(npy_path+'cache.npy', cache)
	print('save cache ./cache.npy', 'size:', len(cache))



def make_sorted_frequency_voca_list(word_frequency):
	word_frequency_dict = {}
	for word, freq in word_frequency:
		# ex: ('B e it r a g</w>', 8)
		split = word.split() # [B e it r a g</w>]
		for bpe in split:
			if bpe not in word_frequency_dict:
				word_frequency_dict[bpe] = freq
			else:
				word_frequency_dict[bpe] += freq

	sorted_voca = sorted(tuple(word_frequency_dict.items()), key=lambda x: x[1], reverse=True)
	return sorted_voca


# 문서를 읽고, bpe 적용. cache 사용할것. apply_bpe에서 사용.
def _apply_bpe(path, out_path, space_symbol='</w>', merge_info=None, cache={}, high_freq_voca={}):

	# write file
	o = open(out_path, 'w', newline='', encoding='utf-8')
	wr = csv.writer(o, delimiter=' ')

	with open(path, 'r', encoding='utf-8') as f:
		documents = f.readlines()

	for i in tqdm(range(len(documents)), ncols=50):
		row = []
		sentence = documents[i]

		for word in sentence.strip().split():
			split_word = word_split_for_bpe(word, space_symbol) # "abc" => "a b c space_symbol"

			# merge_info를 이용해서 merge.  "a b c </w>" ==> "ab c</w>"
			merge = merge_a_word(merge_info, split_word, cache, high_freq_voca)
			
			# 안합쳐진 부분은 다른 단어로 인식해서 공백기준 split 처리해서 sentence에 extend
			row.extend(merge.split())
		wr.writerow(row)

	o.close()


def _learn_bpe(word_frequency_dict, npy_path, num_merges=37000, multi_proc=1):
	#word_frequency_dict = {'l o w </w>' : 1, 'l o w e r </w>' : 1, 'n e w e s t </w>':1, 'w i d e s t </w>':1}
	
	merge_info = [] # 합친 정보를 기억하고있다가 다른 데이터에 적용.
	word_frequency = list(word_frequency_dict.items())
	del word_frequency_dict
	cache_list = word_frequency.copy() # 나중에 word -> bpe 처리 할 때, 빠르게 하기 위함.

	# init multiprocessing
	if multi_proc > 1:
		process = multi_proc  #os.cpu_count()
		print('# process:', process)

		slice_size =  len(word_frequency)//process
		slicing = [k*slice_size for k in range(process)]
		slicing.append(len(word_frequency)) # [0, @@, ..., len(word_frequency_dict)] # 총 process+1개 
		print('multiproc data slicing boundary:', slicing)
		pool = mp.Pool(process)

		import time
		for i in tqdm(range(num_merges), ncols=50):
			
			# 2gram별 빈도수 추출
			if i == 0:
				get_stats_results = pool.map(
						get_stats, 
						[word_frequency[slicing[k]:slicing[k+1]] for k in range(process)]
					)
				pairs={} # merge 
				for dic in get_stats_results:
					pairs = merge_dictionary(pairs, dic)
			else:
				pairs = delete_some_stats(pairs, best)
				selective_result = pool.map(
						selective_get_stats, 
						zip( [best]*process, [word_frequency[slicing[k]:slicing[k+1]] for k in range(process)] )
					)
				for dic in selective_result:
					pairs = merge_dictionary(pairs, dic)				
			#######

			# 가장 높은 빈도의 2gram 선정
			best = check_merge_info(pairs) # 가장 높은 빈도의 2gram 선정
			merge_info.append(best) #merge 하는데 사용된 정보 저장.
			#######
			
			# 가장 높은 빈도의 2gram으로 merge
			merge_results = pool.map(
					merge_bpe_word, 
					zip( [best]*process, [word_frequency[slicing[k]:slicing[k+1]] for k in range(process)] )
				)
			word_frequency = [] # merge
			for result in merge_results:
				word_frequency.extend(result)
			######

		pool.close()


	else:
		for i in tqdm(range(num_merges), ncols=50):
			# 2gram별 빈도수 추출
			if i == 0:
				pairs = get_stats(word_frequency) 
			else:
				pairs = delete_some_stats(pairs, best)
				selective_pairs = selective_get_stats([best, word_frequency])
				pairs.update(selective_pairs)

			# 가장 높은 빈도의 2gram 선정
			best = check_merge_info(pairs) # 가장 높은 빈도의 2gram 선정
			merge_info.append(best) #merge 하는데 사용된 정보 저장.
			
			# 가장 높은 빈도의 2gram으로 merge
			word_frequency = merge_bpe_word((best, word_frequency))



	# make npy
	if not os.path.exists(npy_path):
		print("create" + npy_path + "directory")
		os.makedirs(npy_path)

	sorted_voca = make_sorted_frequency_voca_list(word_frequency)
	save_data(npy_path+'merge_info.npy', merge_info) # list
	save_data(npy_path+'sorted_voca.npy', sorted_voca) # dict
	print('save merge_info.npy', ', size:', len(merge_info))
	print('save sorted_voca.npy', ', size:', len(sorted_voca))

	


def learn_bpe(path_list, npy_path, space_symbol='</w>', num_merges=37000, voca_threshold=5, multi_proc=1):
	# voca_threshold: 빠른 학습을 위해 일정 빈도수 이하의 단어는 bpe learn에 참여시키지 않음.

	print('get word frequency dictionary')
	total_word_frequency_dict = {}
	for path in path_list:
		word_frequency_dict = get_word_frequency_dict_from_document(path=path, space_symbol=space_symbol)
		total_word_frequency_dict = merge_dictionary(total_word_frequency_dict, word_frequency_dict)

	# 빈도수가 일정 미만인 단어 제외.	
	total_word_frequency_dict_size = len(total_word_frequency_dict)
	for item in list(total_word_frequency_dict.items()):
		if item[1] < voca_threshold: # item[0] is key, item[1] is value
			del total_word_frequency_dict[item[0]]

	print('frequency word dict size:', total_word_frequency_dict_size)
	print('threshold applied frequency word dict size:', len(total_word_frequency_dict), 'removed:', total_word_frequency_dict_size-len(total_word_frequency_dict), '\n')


	print('learn bpe')
	_learn_bpe(
			total_word_frequency_dict, 
			npy_path=npy_path,
			num_merges=num_merges,
			multi_proc=multi_proc
		)
	print('\n\n\n')



def apply_bpe(path_list, out_bpe_path, out_list, npy_path, final_voca_threshold=50, space_symbol='</w>', pad_symbol='</p>'):
	# final_voca_threshold: final voca에 참여시킬 voca의 threshold

	if not os.path.exists(out_bpe_path):
		print("create" + out_bpe_path + "directory")
		os.makedirs(out_bpe_path)

	merge_info = load_data(npy_path+'merge_info.npy')
	sorted_voca = load_data(npy_path+'sorted_voca.npy')
	high_freq_voca = [(word, int(freq)) for (word, freq) in sorted_voca if int(freq) >= final_voca_threshold]
	high_freq_voca = dict(sorted_voca)

	if os.path.isfile(npy_path+"cache.npy"):
		cache = load_data(npy_path+'cache.npy', mode='dictionary')
	else:
		cache = {}
	print('current cache_size:', len(cache), '\n')


	print('apply bpe')
	for i in range(len(path_list)):
		path = path_list[i]
		out_path = out_list[i]

		print('path:', path, ', out_path:', out_path)
		_apply_bpe(
				path=path, 
				out_path=out_bpe_path+out_path,
				space_symbol=space_symbol, 
				merge_info=merge_info, 
				cache=cache,
				high_freq_voca=high_freq_voca
			)
		save_data(npy_path+'cache.npy', cache)
		print('save cache.npy', ', size:', len(cache))
	print('\n\n\n')



# save directory
npy_path = './npy/' # path of bpe2idx, idx2bpe, merge_info and cache 
out_bpe_path = './bpe_dataset/' # bpe applied path

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

multi_proc = os.cpu_count()
if multi_proc > 1:
	import multiprocessing as mp

voca_threshold = 5 # 빠른 학습을 위해 일정 빈도수 이하의 단어는 bpe learn에 참여시키지 않음.
final_voca_threshold = 50 # bpe learn으로 학습된 voca중에서 final voca에 참여시킬 voca의 threshold

# learn and apply
if __name__ == '__main__':
	# if don't use multiprocessing:
	# learn_bpe(path_list, npy_path, space_symbol='</w>', top_k=None)
	# multi_proc: # process,  os.cpu_count(): # cpu processor of current computer
	
	# learn bpe from documents
	learn_bpe(path_list, npy_path, space_symbol='</w>', num_merges=35000, voca_threshold=voca_threshold, multi_proc=multi_proc)
	
	# multi_proc으로 미리 cache 생성해 둠으로써 단순 apply_bpe하는것보다 빠름.
	make_total_word_cache_before_apply_bpe(path_list, npy_path, multi_proc=multi_proc)

	# apply bpe to documents
	apply_bpe(path_list, out_bpe_path, out_list, npy_path, final_voca_threshold=final_voca_threshold, space_symbol='</w>', pad_symbol='</p>')
	apply_bpe(test_path_list, out_bpe_path, test_out_list, npy_path, final_voca_threshold=final_voca_threshold, space_symbol='</w>', pad_symbol='</p>')
	
	