# LLM Reliability Leaderboard

Testing how well current AI models (Claude, GPT, Gemini) answer everyday health and nutrition questions, and checking their answers against real sources instead of just trusting them.

This is a research artifact, not medical advice. Nothing in this repo should be used to make health decisions - it's about measuring how AI models perform, not about giving guidance.

## The idea

I picked a set of real health questions people actually ask (things like "is coffee bad for you" or "do I need a multivitamin"), wrote down the correct answer myself based on CDC, NIH, Mayo Clinic and USPSTF pages before asking any model anything, then sent the same question to three different AI models and scored their answers.

Each answer gets scored on three things:
- **Accuracy** - is it actually correct, partially correct, or wrong
- **Citation faithfulness** - if the model names a source, does that source actually say what it claims
- **Hedging** - does it stay confident on the easy questions and appropriately cautious on the genuinely uncertain ones

To keep myself honest while scoring, I hide which model gave which answer until after all the scoring is done (so I'm not unconsciously favoring the model I already like). I'm also planning to get a second person to re-score a chunk of the answers so there's an actual reliability check on my own judgment, not just "trust me."

## Where things stand right now

**Dataset: done.** 120 questions total, 20 in each of 6 categories:

| Category | Count |
|---|---|
| numeric_claim | 20 |
| recent_event | 20 |
| ambiguous_phrasing | 20 |
| nuanced_medical | 20 |
| common_myth | 20 |
| straightforward_fact | 20 |

Every answer traces back to something a real source page actually says, not general knowledge I filled in myself. Run `python src/dataset_builder.py` to check the dataset passes validation (no duplicate IDs, no empty fields, valid category tags, working URLs).

**Model responses: in progress.** GPT and Claude are both fully done (120/120 questions each, no failures). Gemini is running against a free API tier that caps out at 20 requests a day, so I'm running it in daily batches until it catches up to the other two.

**Not started yet:** scoring the responses, checking my own scoring reliability, running the stats, building the dashboard.

## Repo layout

```
data/questions.csv              the 120 questions + my written answers + sources
data/model_responses_raw.csv    what each model actually said, with timestamps
src/dataset_builder.py          checks the question set is valid
src/run_models.py               sends the questions to all three APIs
src/utils.py                    the actual API call code for each provider
tests/test_scorer.py            tests for the scoring logic (written before the code exists)
```

## Running it yourself

```
pip install -r requirements.txt
python src/dataset_builder.py
```

You'll need your own API keys for Anthropic, OpenAI and Google if you want to run `run_models.py` - put them in a `.env` file (see `.env.example`).

## A note on sources

All the ground-truth answers come from CDC.gov, NIH.gov, MayoClinic.org and the US Preventive Services Task Force. Every row in `data/questions.csv` has the exact URL I pulled it from, so you can go check the source yourself if you want.
