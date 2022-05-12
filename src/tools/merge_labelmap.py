import csv
import json

to_merge_file = "/rds/user/wl356/hpc-work/scene_graph_benchmark/datasets/okvqa/VG-SGG-dicts-vgoi6-clipped.json"
to_read_file = '/rds/user/wl356/hpc-work/scene_graph_benchmark/datasets/okvqa/vgcocooiobjects_v1_class2ind.json'
to_write_file = '/rds/user/wl356/hpc-work/scene_graph_benchmark/datasets/okvqa/vgcocooiobjects_v1_merged.json'
with open(to_merge_file,'r') as f:
    data = json.load(f)
    print(data.keys())
    
with open(to_read_file,'r') as f:
    more_labels = json.load(f)

for label_class, label_id in more_labels.items():
    data['label_to_idx'].setdefault(label_class, label_id)
    data['idx_to_label'].setdefault(str(label_id), label_class)

with open(to_write_file, 'w') as f:
    json.dump(data, f)
