import learn_BPE
import apply_BPE
import os

# train data path
path_list = [
		"./dataset/corpus.tc.en/corpus.tc.en", 
		"./dataset/corpus.tc.de/corpus.tc.de"
	] # original data1, data2

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

npy_path = './npy/' 
out_list = ['./bpe_wmt17.en', './bpe_wmt17.de'] # bpe_applied_data1, data2
out_bpe_path = './bpe_dataset/' # bpe applied path
voca_npy_path = npy_path+'voca_from_learn_BPE.npy'
voca_npy_path_semi_final = npy_path+'semi_final_voca.npy'
voca_npy_path_final = npy_path+'final_voca.npy'


# multi process
multi_proc = os.cpu_count()
#if multi_proc > 1:
#	import multiprocessing as mp

voca_threshold = 5 # 빠른 학습을 위해 일정 빈도수 이하의 단어는 bpe learn에 참여시키지 않음.
final_voca_threshold = 50 # bpe learn으로 학습된 voca중에서 final voca에 참여시킬 voca의 threshold

# learn and apply
if __name__ == '__main__':
	# learn bpe from documents
	# learn_bpe 목적은 voca를 구하는것.
	learn_BPE.learn_bpe(
			path_list=path_list, 
			npy_path=npy_path, 
			space_symbol='</w>', 
			num_merges=35000, 
			voca_threshold=voca_threshold, 
			multi_proc=multi_proc
		)
	# if don't use multiprocessing:
	# learn_bpe(path_list, npy_path, space_symbol='</w>', top_k=None)
	# multi_proc: # process,  os.cpu_count(): # cpu processor of current computer



	# bpe 적용하고 모든 bpe 단어 빈도수대로 추출 
		# 기존의 learn_BPE에서 생성된 voca의 freq와 다른 freq의 voca가 생성됨.(apply_BPE의 merge 방식이 learn_BPE의 merge_info 순서대로 하지 않기 때문임.)
	apply_BPE.apply_bpe(
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
	apply_BPE.apply_bpe(
			path_list=path_list, 
			out_bpe_path=out_bpe_path, 
			out_list=out_list, 
			voca_npy_path=voca_npy_path_semi_final, 
			new_voca_npy_path=voca_npy_path_final, 
			final_voca_threshold=final_voca_threshold, 
			space_symbol='</w>'
		)

	# testset bpe apply
	apply_BPE.apply_bpe(
			path_list=test_path_list, 
			out_bpe_path=out_bpe_path, 
			out_list=test_out_list, 
			voca_npy_path=voca_npy_path_final, 
			new_voca_npy_path=None,
			final_voca_threshold=final_voca_threshold, 
			space_symbol='</w>'
		)

