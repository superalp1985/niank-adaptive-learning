# Decision before confirmation

Date: 2026-10-12. No final data have been generated or opened.

Retain all 12 V7 development runs, including low-volume failures and ordinary
history failures. Freeze the code after adding stage/accounting controls only.

At 2,048 histories x 16 cycles (65,536 presentations), seed-11 development:

| Mode | Action success | Risk fine-sign gain | Safe spurious velocity change | SQLite recovery gain |
| --- | ---: | ---: | ---: | ---: |
| Ordinary, plain | 94.14% | 0.1574 | 0.0327 | -0.39 pp |
| Ordinary, event gate | 93.75% | 0.0040 | 0.0020 | 0 pp |
| Paired states, plain | 94.14% | 0.5415 | 0.0057 | 5.08 pp |
| Paired states, event gate | 98.44% | 0.8350 | 0.0103 | 6.25 pp |

These results justify fresh confirmation, not a success claim. Plain paired
states sometimes outperform event gates on response gain in the 512-history
screen. There is a tradeoff, so both must be retained.

Final decision: all four modes at high volume, paired modes at both low-volume
conditions; three optimization seeds, new shared episode IDs, fixed 0.005
reference-use cost. No change to model behavior after final results are opened.
V8 is an independently recorded development branch, not a final-driven patch.
