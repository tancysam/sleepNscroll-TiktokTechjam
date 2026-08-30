# Score-improvement report chart contracts

## Frozen temporal-fold primary deltas

- Claim: no inspected candidate clears the strict +0.002 primary-delta gate on both reused temporal folds; the faithfully reproduced three-member rank portfolio is closest.
- Form: grouped bar chart.
- Encoding: candidate or diagnostic on x, candidate-minus-matched-control primary delta on y, temporal fold by color.
- Source: immutable scientific-record JSON plus the train-only three-member diagnostic.
- QA: both folds remain visible for every method; diagnostic evidence is explicitly labeled; values are raw score deltas, not percentages.

## Objective-disagreement metric components

- Claim: the best worst-fold recorded candidate improves GAUC more than nDCG@5, leaving top-five ranking as the smaller component gain.
- Form: grouped bar chart.
- Encoding: GAUC or nDCG@5 on x, matched-control delta on y, temporal fold by color.
- Source: Attempt 16 immutable scientific records.
- QA: component deltas are not averaged; both folds use the same scale and matched controls.

## Scenario-tab long-view rates

- Claim: scenario prevalence is heterogeneous enough to justify a controlled shared-versus-tab-expert experiment.
- Form: bar chart.
- Encoding: tab code on x, train long-view rate on y; tooltip includes row count and training-row share.
- Source: read-only train-member signal audit.
- QA: rates are fractional inputs formatted as percentages; the chart is descriptive and does not imply causal tab effects.

## Strict-past user-history coverage

- Claim: bounded user-sequence context is available for most training impressions, including 62.7% with at least twenty prior events.
- Form: bar chart.
- Encoding: minimum prior-event count on x, share of training rows on y.
- Source: read-only train-member signal audit.
- QA: equal timestamps are treated as simultaneous; the chart does not claim query-period outcome availability.

## General visual QA

- Every chart has a native semantic table fallback and source affordance.
- Titles state the measured relationship, not a guaranteed future outcome.
- The report distinguishes recorded candidates, train-only diagnostics, and selected final artifacts.
- Quantitative claims retain exact values in datasets and tooltips.
- Colors distinguish folds only; they do not encode good versus bad.
