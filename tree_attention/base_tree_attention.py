import numpy as np
import sys
# Change this to the path of BINFLASH or just to '..' to point to the parent directory
# sys.path.append('/home/agniv/Documents/BINFLASH/')
sys.path.append('..')
from utils import plot_binary_matrix

def get_total_nodes(num_heads, num_predictions):
    total_nodes = 0
    for i in range(num_heads + 1):
        total_nodes += num_predictions ** i
    return total_nodes

def create_tree_mask(num_heads, num_predictions):
    total_nodes = get_total_nodes(num_heads, num_predictions)
    tree_mask = np.zeros((total_nodes, total_nodes))
    for i in range(total_nodes):
        # Self node
        tree_mask[i,i] = 1
        # Parent node
        tree_mask[i, (i -1) // num_predictions] = 1
    return tree_mask[1:,1:]

def patched_tree_mask(num_heads, num_predictions, max_size): # Max nodes will be multiple of 2
    single_tree_mask = create_tree_mask(num_heads, num_predictions)
    total_nodes = single_tree_mask.shape[0]
    # ## Calculation for batch size as a power of 2
    # max_possible_batches = max_size // total_nodes
    # batch_size = 2 ** int(np.log2(max_possible_batches))

    patched_tree_mask = np.zeros((max_size, max_size))
    
    for i in range(0, max_size, total_nodes):
        if i + total_nodes > max_size:
            break
        patched_tree_mask[i:i+total_nodes, i:i+total_nodes] = single_tree_mask
    return patched_tree_mask



def get_nctx_from_batchSize(batch_size, num_heads, num_predictions):
    total_nodes = get_total_nodes(num_heads, num_predictions)
    total_nodes = total_nodes - 1 # Remove the root node as it is absent in the mask
    batched_total_nodes = total_nodes * batch_size
    nctx = 2 ** np.ceil(np.log2(batched_total_nodes)).astype(int)
    return nctx

def get_nctx_from_predictions(num_heads, num_predictions):
    total_nodes = get_total_nodes(num_heads, num_predictions) - 1
    nctx = 2 ** np.ceil(np.log2(total_nodes)).astype(int)
    return nctx

def padded_tree_mask(num_heads, num_predictions):
    n_ctx = get_nctx_from_predictions(num_heads, num_predictions)
    unpad_tree_mask = create_tree_mask(num_heads, num_predictions)
    padded_tree_mask = np.zeros((n_ctx, n_ctx))
    padded_tree_mask[:unpad_tree_mask.shape[0], :unpad_tree_mask.shape[1]] = unpad_tree_mask
    return padded_tree_mask


# for i in range(3,8):
#     test_nctx = get_nctx_from_predictions(i,4)
#     print(f"num_predictions: {i}, nctx: {test_nctx}")

# test_mask, batch_size = patched_tree_mask(7,4, 32768)
# print(test_mask.shape, batch_size)

    