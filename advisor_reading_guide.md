# 給老師看的 Reading Guide（第一次）

目標：先對齊**主軸與範圍**，不是展示完成度。請老師在「結構 + 兩個錨點結果」層級給回饋，
細節章節等方向確認後再打磨。建議閱讀量約 12–15 頁。

---

## Tier 1 — 先看「骨架」（5 分鐘就能掌握全貌）

| 看什麼 | 位置 | 想請老師回饋 |
|--------|------|--------------|
| Abstract | `front/abstract.tex`（正文前，羅馬頁碼） | 這個一段式總結，貢獻的取捨對不對？ |
| §1.3 Contributions | `contents/chapter01.tex` · **p4–6** | 四點貢獻（data scaling / GFM fine-tuning / tokenization / hierarchical+sample-level）的**優先順序與份量**合理嗎？ |
| §1.4 Thesis Organization | **p6** | 章節骨架（Ch4 是主體）認可嗎？ |

---

## Tier 2 — 兩個最扎實、最該被評的「錨點結果」

| 看什麼 | 位置 | 重點數字 / 想請老師回饋 |
|--------|------|--------------------------|
| **Data Scaling** | §4.5 **p47**；分析 §4.5.3 **p51**；Table 4.6 (p48)、Table 4.10 (p51)、Fig 4.3 (p52) | 500K→5M +7.76pp、5M→50M +4.02pp、50M=**67.07%**，clear diminishing returns。這是主貢獻——呈現方式 OK 嗎？ |
| **Tokenization Ablation** ⭐ | §4.11 **p62**；Table 4.18/4.19 (p64) | MT 13-mer **87.42%** vs NT-Genus **67.07%**：從零訓練、小 100× 的模型海放預訓練 GFM。**這題最需要老師定調**：要寫成「NT-v2 的侷限」還是「tokenization 的勝利」？整本論述會跟著轉。 |
| Backbone Ablation | §4.10 **p59**；Table 4.17 (p61)、Fig 4.5 (p63) | NT-v2 vs shallow Transformer，+13.19pp，乾淨的控制實驗，佐證 pre-training 價值。 |

---

## Tier 3 — 展現嚴謹（可口頭帶過或附上）

| 看什麼 | 位置 | 為何值得提 |
|--------|------|------------|
| Data-Leakage Audit | §4.12.4 **p73**；乾淨集 hierarchical §4.12.5 **p74** | 自己抓到 100K test 與訓練集 >99% 重疊，另建 disjoint 測試集重評（66.6%→**66.1%**，僅差 0.5pp），證明數字可信、模型沒有死背 reads。 |

---

## 先別主打（第一次別端出來，等主軸定了再談）

- **258M scaling**：§4.8.3 (p57) + Ch5 未完成段——目前是「未完成 + 82–90% 推估」。本週的 40–45% 是
  bug 產物（用錯 3507sp 資料 + reader 截 60bp），**不是有效數據**，別寫進論文。等修好重跑拿到乾淨數字再更新。
- **Species / Hierarchical**：§4.6、§4.12（15–17%、predicted routing 14.7% < flat baseline）——honest 但偏弱。
- **Sample-level**：§4.13——自己已 caveat「只是 binomial noise stability，不是真實群落泛化」。

---

## 主動丟給老師的 3 個問題

1. **（最重要）** Tokenization 發現怎麼定調？MT 13-mer 87.42% 海放 NT-v2 67.07%——這對「GFM 路線」是反例還是補充？整本論文的 framing 取決於此。
2. 貢獻的優先順序：把「data scaling 是主因」當主貢獻、tokenization 當第二，這樣排對嗎？
3. 範圍收尾：258M 與 species-level 要等乾淨結果補進來，還是以「未來工作」收？這決定我接下來幾週投在哪。
