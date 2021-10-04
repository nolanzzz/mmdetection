import sys
import os
import numpy as np
import json

data_type = sys.argv[1]
out_filename = '../data/MTA_short/{}/annotations_coco.json'.format(data_type)
gt_path = '../data/MTA_short/{}/gt'.format(data_type)

annotations = []
images = dict()
obj_count = 0

for gt_filename in os.listdir(gt_path):
    cam_no = gt_filename[-5]
    print(os.path.join(gt_path, gt_filename))
    gt = np.loadtxt(os.path.join(gt_path, gt_filename), dtype=int, delimiter=',')

    for fid, pid, xtl, ytl, xbr, ybr in gt:
        # Increase frame_id by one to fix difference from gt and image name
        fid += 1

        # Append 9 at beginning to avoid shorter id for cam_0
        # E.g. frame 14 from cam_1 is: 91000014
        image_id = int("9" + cam_no + (str(fid)).zfill(6))
        file_name = "cam_" + cam_no + "_" + (str(fid)).zfill(6) + ".jpg"

        if image_id not in images:
            images[image_id] = dict(
                id=image_id,
                file_name=file_name,
                height=720,
                width=1080)

        bbox = [int(xtl), int(ytl), int(xbr), int(ybr)]
        area = (int(xbr) - int(xtl)) * (int(ybr) - int(ytl))

        poly = [(x + 0.5, y + 0.5) for x, y in zip(range(xtl, xbr), range(ytl, ybr))]
        poly = [p for x in poly for p in x]

        data_anno = dict(
            image_id=image_id,
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

with open(out_filename, 'w') as f:
    json.dump(coco_format_json, f)

# use this command to transform filename first
# for f in * ; do mv -- "$f" "cam_0_$f" ; done