# Python-Byte_Pair_Encoding
Byte Pair Encoding (BPE)


## Env
   * Python 3
   * Numpy 1.15
   * tqdm
   * multiprocessing
   
## Paper
   * Byte-Pair Encoding (BPE): https://arxiv.org/abs/1508.07909  
      
## Code
   * learn_BPE.py
      * functions of bpe learn
   * apply_BPE.py
      * functions of bpe apply
   * run.py
      * run bpe learn and apply
      * MakeFile:
         * npy/
            * merge_info.npy (not used)
            * voca_from_learn_BPE.npy (voca freq from bpe learn)
            * semi_final_voca.[npy, txt] (voca freq from bpe apply with voca_from_learn_BPE)
            * final_voca.[npy, txt] (voca freq from bpe apply with threshold applied semi_final_voca)       
         * bpe_dataset/
            * bpe_applied_data (from final_voca data)
   
## dataset/
   * WMT17 example
   
