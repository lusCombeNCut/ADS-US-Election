from time import time
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet

import torch

import glob
import pandas as pd
import sys

SCRATCH_DIR = "/user/work/ek22528/ads"


def get_outfile(infile):
    if infile.endswith('.gz'):
        infile = infile[:-3]
    return "".join(infile.split("/")[-1])


def process_chunk(infile, outdir, sentiment_analyser, irony_detector):

    df = pd.read_csv(infile, compression='infer', usecols=['id', 'text', 'date'], dtype={'id': str, 'text': str, 'date': str}, index_col='id')

    df['text_cleaned'] = df['text'].apply(lambda tweet: preprocess_tweet(tweet, lang='en'))
    print(f"{infile} preprocessing complete")
    print(torch.cuda.get_device_name())

    start = time()
    df['sentiment'] = sentiment_analyser.predict(df['text_cleaned'])
    df['sentiment'] = df['sentiment'].apply(lambda op: op.output)
    print(f"sentiment anlaysis complete in {(time() - start):2f}", flush=True)

    start = time()
    df['irony'] = irony_detector.predict(df['text_cleaned'])
    df['irony'] = df['irony'].apply(lambda op: op.output)
    print(f"irony anlaysis complete in {(time() - start):2f}", flush=True)

    df.to_csv(f"{outdir}/out_{get_outfile(infile)}")


with open("tf.txt", 'r') as tf:
    toskip = tf.read().strip().split('\n')
    toskip = list([f"{outf[4:]}.gz" for outf in toskip])
    print(toskip, flush=True)

    chunks = glob.glob(f"{SCRATCH_DIR}/usc-x-24-us-election/*.csv.gz")

    sentiment_analyser = create_analyzer(task="sentiment", lang='en')
    irony_detector = create_analyzer(task="irony", lang='en')

    sentiment_analyser.model.to(f"cuda:{sys.argv[2]}")
    irony_detector.model.to(f"cuda:{sys.argv[2]}")

    print(next(sentiment_analyser.model.parameters()).device)
    print(next(irony_detector.model.parameters()).device)

    print(f"{len(chunks)} chunks found", flush=True)
    for c in chunks:
        if c.split('/')[-1] in toskip:
            print("SKIPPING ", c, flush=True)
            continue
        else:
            try:
                process_chunk(c, f"{SCRATCH_DIR}/remaining", sentiment_analyser, irony_detector)
            except Exception as e:
                print(f"WARNING chunk {c} not processed due to {e}", flush=True)
