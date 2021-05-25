import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
print(this_dir)

# Add qlearn to PYTHONPATH
qlearn_path = osp.join(this_dir, '..')
add_path(qlearn_path)

qlearn_path = osp.join(this_dir, '..', 'gym-minigrid')
add_path(qlearn_path)
