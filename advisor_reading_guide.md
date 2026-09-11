# Reading Guide (oral · M2Lab base)

Spine: tokenization bottleneck; depth explains old +13.19; pre-training = data efficiency, not 50M ceiling.
Base: m2lab-sep-147 (~147pp), not PR ~119pp line.

## Tier 1
- Abstract + §1.3 contributions order

## Tier 2 anchors
- Depth 1->29: +15.60 pp (53.92% to 69.52%); +13.19 was depth, not pre-training
- 50M matched depth: 16L 67.94% / 29L 69.52% vs NT-v2 67.08%
- Crossover at low data (0.5M/1M/5M); soil reproduces
- Tokenizer swap >> depth (exact/hashed 13-mer, MT 13-mer)
- 6-mer saturates after 50M; 13-mer keeps rising

## Tier 3
- leakage / subgenus / Bracken; off-catalogue ANI; METAGENE-1/GenomeOcean not tested

## Do not claim
- +13.19 or +19.29 as isolated pre-training
- soil +10 as pre-training if shallow control
- Fix slides if they still say Pre-training: +13.19 pp
