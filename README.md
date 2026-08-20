# selfie_taboo

## Research questions

These are the standing high-level questions for this project. They should stay
relatively stable; treat any paraphrase of them in a plan or design doc as
suspect unless it's a verbatim quote or an explicit narrowing agreed with the
user.

1. Will a SelfIE adapter correctly uncover something the model is actively
   hiding?
2. How does performance differ if the model is control / prompted / fine-tuned
   to complete its task?
3. What if the adapter is trained on the model being tested (as might be more
   realistic if we're trying to use this to detect a misaligned model),
   instead of the base model?

Implementation-level plans live in `plans/`. Those documents should link back
to (or quote) the specific question(s) above they're addressing rather than
restating the research goal in their own words.
