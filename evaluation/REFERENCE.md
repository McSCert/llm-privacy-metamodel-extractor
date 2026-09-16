# Reference point — A2 complete

Config : gpt-4o-2024-11-20, temperature 0.0, seed 12345, n=3
Repo   : tag a2-absence
Gold   : pipeda_gold_standard.xlsx (v1 labels, 1b not yet merged)

Macro F1  41.5%  (39.8 – 43.6)
Micro F1  63.0%  (60.0 – 66.7)
Mean F1   40.4%  (38.8 – 42.5)

Noise floor (n=3): 0.0 points for all concepts EXCEPT
ProcessingActivity_action, which ranges 30.0 points
(4.1, 4.6, 4.10 alternate Use/Collect).
=> Treat ProcessingActivity_action as unreliable until 1b settles its gold labels.
=> Structured enum labels are near-deterministic; free-text fields are not
   (70% of statements byte-identical, ~100% of enum labels stable).

A2 accepted on mechanism evidence, not score delta:
 - chunker: 4.3.8, 4.5.3, 4.7.x, 4.8.3, 4.9.x, 4.10.4 reachable for the first time
 - absence: first non-zero TN counts; Accuracy emitted for the first time;
   enum false positives roughly halved
Earlier runs used the floating `gpt-4o` alias and are NOT comparable to these.
