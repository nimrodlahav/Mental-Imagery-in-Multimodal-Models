def text_inputs(rows, processor, device):
    prompt = [tpl[2] for tpl in rows]
    return processor(text=prompt, return_tensors="pt", padding=True).to(device)

def image_inputs(rows, processor, device):
    pictures = [tpl[1] for tpl in rows]
    prompts = ["USER:<image>\nASSISTANT:" for _ in pictures]
    return processor(text=prompts, images=pictures, return_tensors="pt").to(device)
