# The earnings forecast task

How the `earnings_forecast` prompt task is scored, so you know what your
model is judged on.

## What the earnings forecast task is

`earnings_forecast` is a prompt task. It starts with prompt cycle 1406.

Each prompt names a company and a quarter and asks your model for the
year-over-year percent change in that company's earnings per share (EPS) for
that quarter. A made-up example of the kind of question asked:

> Brightmere Foods is about to report its third-quarter results. Estimate the
> year-over-year % change in its EPS for the quarter.

Your model's forecast is compared with the change the company actually
reported. The closer models rank higher, and the score comes from that rank,
not from the size of the error.

Answers are generated with a 2048-token limit and with thinking off. An
answer that runs past the limit is cut off, and a cut-off answer can lose its
forecast line, which scores 0.

## The required FORECAST line

The last non-empty line of every answer must match this format:

```
FORECAST: +x.x
FORECAST: -x.x
```

That is: the word `FORECAST` in capitals, a colon, one space, a `+` or `-`
sign, 1 to 4 digits, a decimal point, exactly one digit after it, and
nothing else on the line. As a regular expression:

```
^FORECAST: [+-]\d{1,4}\.\d$
```

The number is the percent change: `FORECAST: +15.0` means EPS up 15.0 % from
the same quarter a year earlier, and `FORECAST: -3.2` means down 3.2 %.

`FORECAST:` must appear nowhere else in the answer: not in the reasoning,
not quoted, not in an earlier draft of the line. Blank lines after the
forecast line are fine; any other text after it is not.

An answer scores 0 for that prompt and takes no rank if:

- its last non-empty line is not a valid forecast line, including when text
  follows the forecast line or the answer was cut off before it;
- `FORECAST:` appears anywhere else in the answer;
- the forecast line is in the wrong format.

### Pass and fail examples

| Line | Result | Why |
|---|---|---|
| `FORECAST: +15.0` | pass | |
| `FORECAST: -3.2` | pass | |
| `FORECAST: +0.0` | pass | |
| `FORECAST: +1250.0` | pass | 4 digits before the point is the maximum |
| `15.0` | fail | no `FORECAST:` label and no sign |
| `FORECAST: 15.0` | fail | no sign |
| `FORECAST: +15` | fail | no decimal |
| `FORECAST: +15.00` | fail | two decimals |
| `FORECAST: +15.0%` | fail | extra character after the number |
| `Forecast: +15.0` | fail | label must be capitals |
| `FORECAST:+15.0` | fail | no space after the colon |

## How a forecast is ranked and scored

For each prompt, every model with a valid FORECAST line is ranked by its
error:

```
error = |forecast − actual EPS % change|
```

The actual change is the company's reported result for that quarter. The
smallest error is rank 1.

Points come from the rank:

| Rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 and below |
|---|---|---|---|---|---|---|---|---|---|---|
| Points | 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |

The prompt score is points / 9, so it runs from 0 to 1. A model with no
valid FORECAST line scores 0 and takes no rank, so it does not push anyone
down.

**Ties** share the best rank they span. If two models have the same error
and would take ranks 2 and 3, both get rank 2, which is 8 points (a score of
8 / 9), and the next model is rank 4. Models tied for the closest forecast
all get rank 1 and the full 9 points. This tie rule applies from prompt
cycle 1409; cycles 1406 to 1408 gave ties the average of the ranks they span.

**Identical models** (the same model hash) share one place. A copy of
another miner's model gets the same rank and score as that model and does
not take an extra place, so it does not push other models down.

The cycle score for the task is the average of your prompt scores over all
the cycle's earnings forecast prompts.

### A worked example

The made-up company Brightmere Foods reports EPS up 10.0 % on the year. Four
models answer:

| Model | Forecast | Error | Rank | Points | Score |
|---|---|---|---|---|---|
| A | `FORECAST: +9.0` | 1.0 | 1 | 9 | 1.00 |
| B | `FORECAST: +12.0` | 2.0 | 2 | 8 | 0.89 |
| C | `FORECAST: +8.0` | 2.0 | 2 | 8 | 0.89 |
| D | `FORECAST: +10.0%` | — | none | 0 | 0 |

D's forecast was the closest, but its line has a `%` sign, so it is not a
valid forecast and scores 0.

## How to score well on the earnings forecast task

**End every answer with a well-formed FORECAST line.** A malformed or
missing line scores 0 however good the reasoning is. Check the exact format:
capital `FORECAST`, a colon and a space, a `+` or `-` sign, 1 to 4 digits,
one decimal, and nothing else on the line, as in `FORECAST: +15.0`. Make it
the last line of the answer, write nothing after it, and don't write
`FORECAST:` anywhere else.

**Fit the reasoning inside the token limit.** Answers are generated with a
2048-token limit and with thinking off. If the answer is cut off before the
forecast line, it scores 0. Keep the reasoning short enough that the
forecast line always gets written.

**Give calibrated estimates.** Scoring is by rank among the models, and
anything outside the top 9 scores 0. A huge miss therefore costs no more
than finishing 10th: there is no extra penalty for being far off, and no
extra reward for being exact beyond taking rank 1. What earns points is
being closer than most other models, prompt after prompt.
