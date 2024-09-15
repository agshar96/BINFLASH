import sys
# Change this to the path of BINFLASH or just to '..' to point to the parent directory
# sys.path.append('/home/agniv/Documents/BINFLASH/')
sys.path.append('..')

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from utils import plot_binary_matrix

def init_tokenizer():
    ds = load_dataset("tatsu-lab/alpaca")

    # Initialize a tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return ds, tokenizer

def get_relevant_fields(entry, is_output = False, is_full = False):
    if is_full:
        return f"{entry['instruction']} {entry['input']} {entry['output']}"
    if is_output:
        return f"{entry['output']}"
    return f"{entry['instruction']} {entry['input']}"

def prefix_llm_mask(maskMat, row_start, col_start, length_r, length_c, is_tril = False):
    if is_tril:
        maskMat[row_start:row_start+length_r, col_start:col_start+length_c] = np.tril(np.ones((length_r, length_c)))
    else:
        maskMat[row_start:row_start+length_r, col_start:col_start+length_c] = 1
    return maskMat

def return_prefix_llm_mask(max_length):
    ds, tokenizer = init_tokenizer()

    attn_mask_mat = np.zeros((max_length, max_length))
    row_start = 0
    col_start = 0
    for i in range(len(ds['train'])):

        cur_entry = ds['train'][i]
        input_entry = get_relevant_fields(cur_entry)
        tokenized_entry = tokenizer(input_entry, return_tensors='pt')
        input_ids = tokenized_entry['input_ids']

        output_entry = get_relevant_fields(cur_entry, is_output=True)
        tokenized_output = tokenizer(output_entry, return_tensors='pt')
        output_ids = tokenized_output['input_ids']

        total_length = input_ids.shape[1] + output_ids.shape[1]

        if row_start + total_length <= max_length and col_start + total_length <= max_length:
            ## Input bidirectional mask
            attn_mask_mat = prefix_llm_mask(attn_mask_mat, row_start, col_start, input_ids.shape[1], input_ids.shape[1])
            row_start = row_start + input_ids.shape[1]

            ## Output input mask
            attn_mask_mat = prefix_llm_mask(attn_mask_mat, row_start, col_start, output_ids.shape[1], input_ids.shape[1])
            col_start = col_start + input_ids.shape[1]

            ## Output causal mask
            attn_mask_mat = prefix_llm_mask(attn_mask_mat, row_start, col_start, output_ids.shape[1], output_ids.shape[1], is_tril=True)
            row_start = row_start + output_ids.shape[1]
            col_start = col_start + output_ids.shape[1]
        else:
            break

    return attn_mask_mat

# test_mask = return_prefix_llm_mask(1024)
# plot_binary_matrix(test_mask)