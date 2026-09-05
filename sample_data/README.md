# sample_data/ — intentionally empty

**No photographs of real people are committed to this repository.** Face images
are biometric data; publishing someone's photo in a public repo is not something
a test fixture should do, and it is not needed to run or grade this project.

## What to put here

At run time, supply your own test photo and point the pipeline at it:

```bash
python pipeline.py --image sample_data/my_photo.jpg
```

The photo must be:

1. **Of a consenting subject** — yourself, or a teammate who has agreed to it.
2. **Already publicly posted** on a social profile (Instagram, X, LinkedIn, …).
   Stage 2 asks Google what it has already indexed; a photo that has never been
   online cannot be matched, and the pipeline will honestly report
   "no matching social-media post found". See README → *Known limitations*.
3. **A single face**, reasonably front-facing. Stage 1 refuses an image with zero
   or multiple faces rather than guessing which person you meant, so crop group
   photos down to one subject first.

## Do not commit your test photo

Anything you drop in this directory other than this note should stay local. If
you initialise git here, add a `.gitignore` line for it:

```
sample_data/*
!sample_data/README.md
```
