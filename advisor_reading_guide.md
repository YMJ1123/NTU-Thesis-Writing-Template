# 給老師看的 Reading Guide（口試稿）

目標：對齊**可辯護主軸**（tokenization 是瓶頸；深度可彌補不良 tokenizer；預訓練買的是資料效率，不是 50M 上限）。建議先看 Abstract + §1.3，再看 §4.9 深度掃描與 §4.10 同架構 tokenizer 表。

---

## Tier 1 — 骨架

| 看什麼 | 位置 | 想請老師回饋 |
|--------|------|--------------|
| Abstract | `front/abstract.tex` | 主軸與 caveat（leakage、subgenus、屬豐度、METAGENE-1/GenomeOcean 未測）是否正確？ |
| §1.3 Contributions | `contents/chapter01.tex` | 貢獻順序：tokenization → 深度對齊預訓練 → data scaling → 階層/樣本。 |

---

## Tier 2 — 口試會被問的錨點

| 看什麼 | 位置 | 重點數字 |
|--------|------|----------|
| **Depth sweep** | §4.9 Table `tab:depth-sweep` | 1/8/16 層從頭訓練 53.92 / 65.44 / **67.94\%** vs NT-v2 **67.08\%**（同一 `clean_common`）。舊的 +13.19~pp 是深度，不是預訓練。 |
| **5M crossover** | Table `tab:crossover-summary` | 5M：**+8.94~pp** 預訓練；50M：**−0.86~pp**。63.05\% 是內部驗證數字。 |
| **Tokenization** | §4.10；`tab:samearch-tok` | hashed 13-mer **83.83\%**、exact **85.63\%**、MT 13-mer **87.42\%**。6-mer 29 層也只有 69.52\%。 |
| **Data scaling** | §4.5 | 500K→5M +7.76~pp、5M→50M +4.02~pp、50M=**67.07\%**、250M +0.22~pp。13-mer 同區間 87.5→98.7\%。 |

---

## Tier 3 — 嚴謹（口頭可帶過）

- Data-leakage audit §4.11：66.6\%→**66.1\%**。
- Subgenus 負結果；Bracken $r=0.997$；219-species FASTA 缺口。
- 範圍：一般用途 GFM；METAGENE-1 / GenomeOcean **未測**。

---

## 口試不要再主張的句子

- 「+13.19~pp 是分離出來的預訓練效果」（1 層 vs 29 層）。
- 「+19.29~pp 是分離出來的預訓練效果」（MT 1 block vs NT-v2）。
- 「soil 50M +10~pp 是預訓練」（仍是 1-layer MT）。
- 「DNABERT 50M 訓練進行中」。
