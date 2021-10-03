import sys
import numpy as np
import json

# data_type = sys.argv[1]
# cam_id = sys.argv[1]
# path = './cam_1_short.csv'
# filename = path + '/coords_fib_cam_' + folder_no + '.csv'
gt_filename = './coords_fib_cam_1.csv'
out_filename = './cam_1_coco.json'

annotations = []
images = dict()
obj_count = 0

gt = np.loadtxt(gt_filename, dtype=int, delimiter=',')

for fid, pid, xtl, ytl, xbr, ybr in gt:
    fid += 1
    file_name = (str(fid)).zfill(6) + ".jpg"
    # id = "cam_{}_{}".format(cam_id, (str(fid)).zfill(6))
    if fid not in images:
        images[fid] = dict(
            id=int(fid),
            file_name=file_name,
            height=720,
            width=1080)

    bbox = [int(xtl), int(ytl), int(xbr), int(ybr)]
    area = (int(xbr) - int(xtl)) * (int(ybr) - int(ytl))

    poly = [(x + 0.5, y + 0.5) for x, y in zip(range(xtl, xbr), range(ytl, ybr))]
    poly = [p for x in poly for p in x]

    data_anno = dict(
        image_id=int(fid),
        id=obj_count,
        category_id=0,
        bbox=bbox,
        area=area,
        segmentation=[poly],
        iscrowd=0)

    annotations.append(data_anno)
    obj_count += 1

coco_format_json = dict(
    images=list(images.values()),
    annotations=annotations,
    categories=[{'id': 0, 'name': 'pedestrian'}]
)
# print(coco_format_json)
with open(out_filename, 'w') as f:
    json.dump(coco_format_json, f)
