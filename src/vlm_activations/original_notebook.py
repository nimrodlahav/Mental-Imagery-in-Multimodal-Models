from google.colab import drive
from google.colab import files
import os
import random
from pycocotools.coco import COCO
from PIL import Image
from IPython.display import display
from transformers import AutoProcessor, AutoModelForImageTextToText, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch, requests
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import numpy as np
import json

!pip uninstall -y bitsandbytes
!pip install -q "bitsandbytes>=0.41.0" "transformers>=4.41.0" accelerate

drive.mount('/content/drive')

ann_file = "/content/drive/MyDrive/Workshop/annotations/captions_val2017.json"
coco = COCO(ann_file)
all_ids = coco.getImgIds()
random.shuffle(all_ids)
with open("/content/drive/MyDrive/shuffled_ids.json", "w") as f:
  json.dump(all_ids, f)

current_index = 0   # global pointer to the img id index in all_ids

def extract_twenty():
  global current_index
  with open("/content/drive/MyDrive/shuffled_ids.json", "r") as f:
      all_ids = json.load(f)
    # take the next 5 IDs
  sampled_ids = all_ids[current_index:current_index+10]
  current_index += 10  # advance pointer

  base_dir = "/content/drive/MyDrive/Workshop"
  img_dir = os.path.join(base_dir, "val2017")
  ann_file = os.path.join(base_dir, "annotations", "captions_val2017.json")

  # Initialize COCO API
  coco = COCO(ann_file)

  rows_COCO = []
  for img_id in sampled_ids:
      # Load image metadata
      img_info = coco.loadImgs(img_id)[0]
      img_path = os.path.join(img_dir, img_info["file_name"])
      # Load captions
      ann_ids = coco.getAnnIds(imgIds=img_id)
      anns = coco.loadAnns(ann_ids)
      # Open image
      image = Image.open(img_path).convert("RGB")

      # Print/display
      #print(f"Image ID: {img_id} | File: {img_info['file_name']}")
      first_caption = anns[0]["caption"]
      #print(first_caption)
      #display(image)
      #print("-" * 60)

      rows_COCO.append((img_id, image, first_caption))
  return rows_COCO


def load_model():
  model_id = "llava-hf/llava-1.5-7b-hf"
  quant_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype="float16"
  )
  model = LlavaForConditionalGeneration.from_pretrained(
      model_id,
      quantization_config=quant_config,
      device_map="auto"
  )
  processor = AutoProcessor.from_pretrained(model_id)
  return model, processor

def preprocess_text_inputs(rows, processor):
  prompt = [tpls[2] for tpls in rows]    ## all captions as text inputs
  text_only_inputs = processor(text=prompt, return_tensors="pt", padding=True).to(nn_model[0].device)
  return text_only_inputs

def preprocess_image_inputs(rows, processor):
  pictures = [tpls[1] for tpls in rows]        ## all images as image inputs
  image_only_prompt = ["USER:<image>\nASSISTANT:" for image in pictures]
  image_only_inputs = processor(text=image_only_prompt, images=pictures, return_tensors="pt").to(nn_model[0].device)
  return image_only_inputs

def outputs(inputs, model_instance):
  out = model_instance(**inputs,
        max_length=512,
        output_hidden_states=True,
        do_sample=False,
        return_dict=True)
  return out

def collect_activations(global_index, rows):
    # Define schema once
    schema = pa.schema([
        ("value", pa.float32()),
        ("layer", pa.int32()),
        ("batch", pa.int32()),
        ("token", pa.int32()),
        ("neuron", pa.int32()),
        ("input_idx", pa.int32()),
        ("modality", pa.string())
    ])

    filename = f"/content/drive/MyDrive/activations_batch_{global_index}.parquet"
    writer = pq.ParquetWriter(filename, schema)

    def make_hook(layer_idx, typ):
        def hook(module, input, output):
            layer_act = output.detach().cpu().numpy()
            batch_size, seq_len, hidden_dim = layer_act.shape

            # Instead of saving in memory, create an Arrow Table per layer
            entries = []
            for b in range(batch_size):
                for t in range(seq_len):
                    for neuron_idx, val in enumerate(layer_act[b, t]):
                        entries.append({
                            "value": float(val),
                            "layer": layer_idx,
                            "batch": b,
                            "token": t,
                            "neuron": neuron_idx,
                            "input_idx": global_index * batch_size + b,
                            "modality": typ
                        })
            table = pa.Table.from_pylist(entries, schema=schema)
            writer.write_table(table)  # write directly to disk
        return hook

    for typ in ["text", "vision"]:
        hooks = [
            layer.mlp.act_fn.register_forward_hook(make_hook(i, typ))
            for i, layer in enumerate(nn_model[0].model.language_model.layers)
        ]

        if typ == "text":
            inputs = preprocess_text_inputs(rows, nn_model[1])
        else:
            inputs = preprocess_image_inputs(rows, nn_model[1])

        _ = outputs(inputs, nn_model[0])

        for h in hooks:
            h.remove()

    writer.close()

# run code
nn_model = load_model()
for i in range(21,100):
    batch = extract_twenty()
    table = collect_activations(i, batch)
    filename = f"/content/drive/MyDrive/activations_batch_{i}.parquet"
