import os
import genomepy
import pandas as pd
from torch.nn import functional as F
import torch
import numpy as np
data = pd.read_table("pchic/0071_Blood - Non-activated CD4+ Cells_merged_loop.txt")
standard_size = 5_000
def expand_enrich_pchic(df):
    df[["bait_frag_chr", "bait_frag_start", "bait_frag_end"]] = df["bait_frag"].str.split(",", expand=True)
    df[["other_frag_chr", "other_frag_start", "other_frag_end"]] = df["other_frag"].str.split(",", expand=True)
    df["bait_size"] = np.abs(df["bait_frag_start"].apply(int) - df["bait_frag_end"].apply(int))
    df["other_size"] = np.abs(df["other_frag_start"].apply(int) - df["other_frag_end"].apply(int))
    df["bait_frag_midpoint"] = df.apply(lambda x: int((int(x["bait_frag_start"]) + int(x["bait_frag_end"])) / 2), axis=1)
    df["bait_frag_standard_size_start"] = df["bait_frag_midpoint"] - standard_size
    df["bait_frag_standard_size_end"]   = df["bait_frag_midpoint"] + standard_size
    return df
expand_enrich_pchic(data)
genomepy.install_genome("hg19")
hg = genomepy.Genome("hg19")
def fasta_to_ohe_mapping(ascii_dna):
        if ascii_dna == 65:  # A
            return 1
        if ascii_dna == 67:  # C
            return 2
        if ascii_dna == 71:  # G
            return 3
        if ascii_dna == 84:  # T
            return 4
        if ascii_dna == 97:  # a
            return 1
        if ascii_dna == 99:  # c
            return 2
        if ascii_dna == 103: # g
            return 3
        if ascii_dna == 116: # t
            return 4
        return 0 # Nn, anything else

def from_loc_to_ohe(chr, start, end):
    sequence = hg.get_seq(chr, start, end)
    buffer = torch.frombuffer(sequence.seq.encode(), dtype=torch.uint8)
    buffer = buffer.apply_(fasta_to_ohe_mapping)
    atcg_mask = buffer != 0 # 1 if actg, 0 otherwise
    buffer -= atcg_mask * 1 # offset indexing for atcg so it starts at 0 instead of 1
                            # this is needed to have the "others" be a 0000 row in OHE
    return (
        F.one_hot(buffer.long(), num_classes=4).T * atcg_mask # OHE and mask others
    )
n=1000
chunks = [data[i:i+n] for i in range(0,data.shape[0],n)]
for j, chunk in enumerate(chunks):
    chunk_ohe = []
    for i, row in chunks[0].iterrows():
        ohe = from_loc_to_ohe(row["bait_frag_chr"], row["bait_frag_standard_size_start"], row["bait_frag_standard_size_end"])
        chunk_ohe.append(ohe)
        assert ohe.shape[1] == 10001
        print(f"Chunk {j:>4} - Row {i:>4}", end="\r")
    torch.save(torch.stack(chunk_ohe), f"pchic/intermediates/ohe_chunk_{j}.pt")
