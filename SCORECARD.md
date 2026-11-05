# Scorecard: scenario D, C1 to C8, 3 runs each

Same fictional month on both sides (1,326 screen messages plus Gmail and Calendar connector copies, promos and a stale invite). Reader: Opus via headless claude -p with only that side's connector. Judge: Opus seeing only the question, the rubric in corpus/scenario_c_questions.yaml and the answer. Every failed verdict was audited by hand; the two harness fails are date-precision calls (Diwali deadline given as Sat Nov 14 instead of the weekend; deck sent on WhatsApp without the Nov 13 send date) and are kept as fails.

| Question | System | Passed / judged | Mean tool calls | Mean ≈ tokens read from memory | Mean duration (s) |
|---|---|---|---|---|---|
| C1 | minimi_shape | 0/3 | 2.0 | 2276 | 38.5 |
| C1 | harness | 3/3 | 1.3 | 261 | 27.6 |
| C2 | minimi_shape | 0/3 | 8.7 | 8709 | 44.9 |
| C2 | harness | 2/3 | 1.0 | 640 | 31.7 |
| C3 | minimi_shape | 3/3 | 3.7 | 3802 | 38.9 |
| C3 | harness | 2/3 | 1.0 | 113 | 27.6 |
| C4 | minimi_shape | 3/3 | 4.0 | 1886 | 33.9 |
| C4 | harness | 3/3 | 2.0 | 674 | 30.0 |
| C5 | minimi_shape | 0/3 | 10.0 | 6915 | 51.3 |
| C5 | harness | 3/3 | 4.0 | 5065 | 43.2 |
| C6 | minimi_shape | 2/3 | 3.0 | 2098 | 35.8 |
| C6 | harness | 3/3 | 1.0 | 237 | 33.0 |
| C7 | minimi_shape | 3/3 | 2.7 | 3389 | 35.3 |
| C7 | harness | 3/3 | 2.3 | 1529 | 33.2 |
| C8 | minimi_shape | 3/3 | 5.0 | 5480 | 42.3 |
| C8 | harness | 3/3 | 1.0 | 159 | 42.5 |

| System | Questions with all 3 runs passed | Runs passed / judged | Mean tool calls | Mean ≈ tokens read from memory | Mean duration (s) |
|---|---|---|---|---|---|
| minimi_shape | 4/8 | 14/24 | 4.9 | 4319 | 40.1 |
| harness | 6/8 | 22/24 | 1.7 | 1085 | 33.6 |

- ≈ tokens read from memory: characters of every tool result in a run divided by 4, averaged over runs.
