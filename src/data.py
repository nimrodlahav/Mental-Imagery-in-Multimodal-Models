import json, os, random
from pycocotools.coco import COCO
from PIL import Image

def shuffle_and_save_ids(coco_ann_path: str, out_json: str):
    coco = COCO(coco_ann_path)
    all_ids = coco.getImgIds()
    random.shuffle(all_ids)
    with open(out_json, "w") as f:
        json.dump(all_ids, f)

def load_batch(coco_root: str, shuffled_ids_json: str, start_idx: int, batch_size: int):
    img_dir = os.path.join(coco_root, "val2017")
    ann_file = os.path.join(coco_root, "annotations", "captions_val2017.json")
    coco = COCO(ann_file)

    with open(shuffled_ids_json, "r") as f:
        all_ids = json.load(f)

    sampled_ids = all_ids[start_idx:start_idx+batch_size]
    rows = []
    for img_id in sampled_ids:
        info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, info["file_name"])
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        image = Image.open(img_path).convert("RGB")
        first_caption = anns[0]["caption"]
        rows.append((img_id, image, first_caption))
    return rows
