from m3inference import M3Inference
import pprint

m3 = M3Inference(use_full_model=False) # see docstring for details

pred = m3.infer('part_1.jsonl')

pprint.pprint(pred)