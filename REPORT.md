# Methodology and findings

Work in progress. This file gets filled in as each phase of the project actually finishes, not written up front and backfilled. Anything marked TBD below hasn't happened yet.

The two things I think matter most about how this is set up, more than the usual "collect data, run stats, make a dashboard" pattern: I score the model answers blind (I don't know which model I'm looking at while scoring), and I'm getting a second person to check a sample of my scoring so there's an actual reliability number attached to it instead of just my word.

## 1. Overview

TBD, will fill in once scoring starts. Will cover the domain choice, the scoring axes, the blind scoring setup, and which exact model versions got tested on which date.

## 2. Building the question set

Done. 120 questions, 20 per category, all in `data/questions.csv`. Run `python src/dataset_builder.py` if you want to check it yourself (checks for duplicate IDs, missing fields, bad category tags, broken URLs).

A few notes on how this actually went:

- I only pulled from CDC.gov, NIH.gov and its subdomains (NIDDK, NHLBI, NIA, NCCIH, ODS), MayoClinic.org (plus its newsnetwork and healthsystem subdomains), and the US Preventive Services Task Force site. Nothing else.
- Every answer had to trace back to something the source page actually says in plain language, not something I know to be generally true. If I couldn't point to a sentence, the question didn't go in.
- I dropped a few candidate questions along the way. One worth mentioning: I had a "vaccines cause autism" myth question planned, but when I went to pull the CDC source, the page's wording had shifted and didn't cleanly debunk the claim anymore on a close read, so I cut it rather than force a shaky citation.
- Still need to do: have someone other than me spot-check a sample of the answers against the sources before I start using this dataset to grade models. Haven't done that yet, worth doing before Phase 2 wraps up.

## 3. Scoring rubric

TBD, this is Phase 3.

Plan is three axes: accuracy (0, 0.5, or 1), citation faithfulness (0 or 1, only applies if the model actually cited something), and hedging (0 or 1). Scoring happens blind, model identity hidden until after. There will also be an AI-assisted first pass on citation checking that I manually verify a sample of before trusting it.

## 4. Checking my own scoring reliability

TBD, hasn't started.

When this happens, I'll report whatever the actual agreement number comes out to be, kappa or percent agreement, even if it's mediocre. A believable number that shows where I was inconsistent is worth more than a suspiciously perfect one.

## 5. Statistical findings

TBD, hasn't started.

Given the dataset size (~360 total responses), I'm not going to run anything fancier than a chi-square test and a logistic regression with confidence intervals reported. If nothing comes out statistically significant after correcting for multiple comparisons, that's a real finding too, not a failure of the project.

## 6. Limitations

What I already know going in:

- 120 questions across 6 categories means about 20 questions per category per model, which is a small sample. Any stats I report will have confidence intervals attached for exactly this reason.
- Ground truth is limited to what CDC, NIH, Mayo Clinic and USPSTF pages state plainly. That's a scope choice, not a claim that this covers all of health topic space.
- Model behavior changes over time even for the "same" model name. Whatever I find only describes the specific pinned versions on the specific dates in `data/model_responses_raw.csv`.
- This is a research project measuring AI model behavior. It is not medical advice, and the dashboard will say so clearly once it exists.

## 7. Disclaimer

This project looks at how AI models answer health-related questions. It is not medical advice. The answers used as ground truth come from CDC, NIH, Mayo Clinic and USPSTF as of whenever I checked each source, and none of it should replace talking to an actual doctor.
