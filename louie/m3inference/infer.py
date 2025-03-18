from m3inference import M3Inference
import pprint

m3 = M3Inference(use_full_model=False) # see docstring for details
# pred = m3.infer('./test/data_resized.jsonl') # also see docstring for details

# pred = m3.infer('aug_chunk_41.jsonl')
pred = m3.infer('small_chunk.jsonl')


pprint.pprint(pred)