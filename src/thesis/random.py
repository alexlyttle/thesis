from jax import tree_util, random

def split_like_tree(key, tree=None, treedef=None):
    if treedef is None:
        treedef = tree_util.tree_structure(tree)
    keys = random.split(key, treedef.num_leaves)
    return tree_util.tree_unflatten(treedef, keys)
