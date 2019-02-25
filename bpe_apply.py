import argparse
import os
import bpe_module.apply_BPE as apply_BPE

parser = argparse.ArgumentParser(description='file path')
parser.add_argument(
		'-data_path', 
		help="Multiple documents path",
		required=True, 
		nargs='+'
	)
parser.add_argument(
		'-voca_path', 
		help="Vocabulary for BPE apply",
		required=True
	)
parser.add_argument(
		'-bpe_out_path', 
		help="Multile BPE_applied path",
		required=True, 
		nargs='+'
	)

args = parser.parse_args()

data_path = args.data_path
voca_path = args.voca_path
bpe_out_path = args.bpe_out_path

if __name__ == '__main__':	

	apply_BPE.apply_bpe(
			path_list=data_path, 
			out_list=bpe_out_path, 
			voca_path=voca_path, 
			space_symbol='</w>'
		)