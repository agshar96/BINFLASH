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

def patched_attention_mask(patch_att_mat, start_location, length, is_output = False, is_full = False):
    if is_output or is_full:
        patch_att_mat[start_location:start_location+length, start_location:start_location+length] = np.tril(np.ones((length, length)))
    else:
        patch_att_mat[start_location:start_location+length, start_location:start_location+length] = np.ones((length, length))
    return patch_att_mat

def create_attention_mask(ds, tokenizer, max_length, is_output = False, is_full = False):

    patched_attn_mat = np.zeros((max_length, max_length))
    cur_start = 0
    for i in range(len(ds['train'])):
        cur_entry = ds['train'][i]
        combined_entry = get_relevant_fields(cur_entry, is_output, is_full)
        tokenized_entry = tokenizer(combined_entry, return_tensors='pt')
        input_ids = tokenized_entry['input_ids']

        ## Now we make the attention matrix
        if cur_start + input_ids.shape[1] <= max_length:
            patched_attn_mat = patched_attention_mask(patched_attn_mat, cur_start, input_ids.shape[1], is_output, is_full)
        else:
            break
        cur_start = cur_start + input_ids.shape[1]

    return patched_attn_mat

def return_full_alpaca_mask(max_length):
    ds, tokenizer = init_tokenizer()
    patched_attn_mat = create_attention_mask(ds, tokenizer, max_length, is_full=True)
    return patched_attn_mat
# # Set the maximum length of the sequence
# max_length = 1024
# patched_attn_mat = create_attention_mask(ds, tokenizer, max_length)
# plot_binary_matrix(patched_attn_mat)