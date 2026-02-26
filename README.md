# Bardic Fine-Tuning

## Quickstart
- Install toolchain (macOS): `uv venv && source .venv/bin/activate && uv pip install --upgrade pip`
- Install deps: `uv pip install -e .`
- Launch Jupyter: `uv run python -m ipykernel install --user --name bardic-env --display-name "bardic-env"`
- Start notebooks: `uv run jupyter lab`

## Notebooks
- 01_Data_Generator.ipynb — Generate/refresh `data/poetic_refusal.jsonl` via OpenAI/Anthropic or stubbed bardic refusals.
- 02_Trainer_Arena.ipynb — Sequential LoRA then DoRA fine-tunes with Unsloth, saving adapters under `outputs/`.
- 03_Judge_Arena.ipynb — Load both adapters, run side-by-side refusals, and compare outputs.

## Data
- Seed file: `data/poetic_refusal.jsonl` with system/user/assistant rows in chat format.

## Notes
- Core stack: unsloth, transformers, trl, peft, datasets, torch, accelerate, wandb (optional), ipywidgets.
- Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in your env to hit live providers in the data generator.


# BASE MODEL BEHAVIOUR, no FT
=== no system prompt ===
- sometimes like a normal chat, nothing new
- sometimes doesn't respond at all
- sometimes it even spits out the supposed system prompt LMAO

=== system prompt ===
- Resonds in like a role-playing style
- Doesn't really follow the format that i want it to follow

# good-enough/checkpoint-860: mistralInstruct, LoRa 6.8k samples, 32/64, 1 epoch, 2e-4 lr, linear scheduler, alpaca/instruct
- Was not given full details of what to take note of since its the first successful-ish run

# good-enough/mistral-025-best: mistralInstruct, LoRa 31k samples, 32/64, 1 epoch, 2e-4 lr, linear scheduler, conversation format
- it does follow the format that was trained on with Context: ... then follows up with the poem
- but half the time it breaks, either outputting nth, a random number, some chinese char
- loss 0.3618 which is p good but performance is otherwise
- tested with recommended mistral values and no system prompt

# good-enough/checkpoint-2940: mistralBase, LoRa 24k samples, 32/64, 1 epoch, 2e-4 lr, cosine scheduler, conversation format
=== no system prompt ===
- feels less poetic
- more occurences of not direct poem replies

=== with system prompt ===
- the poems actually need to think a bit one
- replies directly with one single poem MOST of the time 9/10 kind

# dora_runs/checkpoint-770-conv-resp-only-bs16: mistralBase, DoRa 12k samples, 32/64, 1 epoch, 2e-4 lr, cosine scheduler, conversation format responses only
- it didn't generate SHIT, with or without system prompt
- never again

# good-enough/checkpoint-850-conv-bs8, DoRa 6.8k samples, 32/64, 1 epoch, 2e-4 lr, cosine scheduler, conversation format
=== no system prompt ===
- most responses are long, like a normal chat
- when there are poems at the end, they're too direct, super obvious
- feels quite slow also in generation

=== system prompt ===
- all except 1 were direct responses
- feels more like proses than poems
- very simple, straightforward


# unknown exact configuration, assumed to have the following checkpoint-640: mistralBase, LoRa 6k samples, 32/64, 1 epoch, 2e-4 lr, cosine scheduler, conversation format responses only
=== no system prompt ===
- absolutely does not yap in poems
- feels very normal
- sometimes responds properly, sometimes just goes on and on
- TERRIBLE without system prompt

=== system prompt ===
- always one line, gives occasional good poems
- but for some reason, most start with 'no'????
- pretty quick



<!-- THIS IS AFTER THE DATASET UPGRADE -->

# good-enough/checkpoint-125-16_32-cosine: mistralBase, LoRa 1k samples, 16/32, 1 epoch, 2e-4 lr, cosine scheduler, conversation format
=== no system prompt ===
- like a normal chat, in general
- VERY SOMETIMES there's a poem-like thing
- Could leak the training system prompt sometimes??? but when it does, it spits out a poem
- repeats itself a lot

=== system prompt ===
- all responses become actual poems
- but they're super long
- slight variation in length, but generally quite long
- sometimes is stuck in a loop after a few stanzas or much at the end

# good-enough\base-32_64-conv-cosine, mistralBase, LoRa 7k samples, 32/64, 1 epoch, 2e-4 lr, cosine scheduler, conversation format
=== no system prompt ===
- absolutely does not respond in a poetic way, just normally (better than 1k samples)
- But at least it responds, unlike before finetuned
- it starts to repeat itself after a while quite often (not as bad as 1k samples)

=== system prompt ===
- slightly more than half the responses are acceptably poetic
- Lengths are generally quite long
- There is some variation in length (more variance than 1k samples)

# good-enough\checkpoint-125-32_64-cosine, mistralBase, LoRa 1k samples, 32/64, 1 epoch, 2e-4 lr, cosine scheduler, conversation format
> this is to test if the 16/32 rank/alpha is the factor. if results are similar to 16/32, then we know that 32/64 is not the reason for the better qualitative performance, and its to do with overfitting
>> so it seems that the rank/alpha is not the main factor, and it's more to do with overfitting
>> therefore, just stop early, and use higher rank/alpha
=== no system prompt ===
- in general sounds like a normal chat
- occasionally, there's poetic-ish responses, but i wouldn't call them actual poems (short sentences, with many commas)
- it doesn't do the new lines thing
- repeats itself a lot

=== system prompt ===
- quite long poetic responses
- but there are more shorter ones than normal
- sometimes things get repeated, but not too bad
- they don't feel like there's much analysis needed
- but they feel comfortable to read
- id say higher quality overall

# good-enough\base-32_64-conv-cosine-fast-learner, mistralBase, LoRa 7k samples, 32/64, 1 epoch, 5e-4 lr, cosine scheduler, conversation format
=== no system prompt ===
- either you get poetic responses, system prompt leak, or massive repetition
- leaks trained system prompt??? then spits out a totally unrelated poem
- repeats itself a lot

=== system prompt ===
- long poetic responses
- no short responses
- rarely do things get repeated
- honestly quite good ones in there, non-direct, with some analysis
- BUT if it comes across things that it can't generate, it goes TERRIBLY (repeating itself too much, yapping nonsense)
- Edge case of it responding like normal actually

# good-enough/dropout-full-cp80-val-loss-best-1k, mistralBase, LoRa 1k samples, 32/64, dropout 5%, 2e-4 lr, cosine scheduler, conversation format
> dropout doesn't seem to be useful to prevent overfitting, or is it because its only 1k samples that it doesn't rlly matter
=== no system prompt ===
- absolutely fking useless
- i don't even want to have this conversation

=== system prompt ===
- around 1/4 are not prose-like or poetic
- half of the rest are slightly poetic, not v high quality
- the rest are just proses, not really deep
- sometimes repeats itself

# good-enough/dropout-full-cp310-val-loss-shite, mistralBase, LoRa 7k samples, 32/64, dropout 5%, 2e-4 lr, cosine scheduler, conversation format
> because in this case, it rlly COOKED well with dropout... so what now brah
> can't really rely on the validation loss in this case. took a random checkpoint.
=== no system prompt ===
- literally keeps repeating itself like non stop
- very incoherent
- only 1/10 makes sense, but its not high quality
- goes to the max often

=== system prompt ===
- 1/10 were broken
- the rest are legitimately quite good
- Varying lengths of poems as well
- its really damn good though...


# good-enough/dora-1k/dora_adapter, mistralBase, DoRa 1k samples, 32/64, 2e-4 lr, cosine scheduler, conversation format
=== no system prompt ===
- literally keeps repeating itself like non stop
- very incoherent
- goes to the max often

=== system prompt ===
- dude. 
- its really really damn good this time LMAO
- varying lengths, great quality, non-direct with some analysis
- parts actually rhyme most of the time
- only 1/10 is not great, but still good compared to the rest

# good-enough/dora-response-only/dora-adapter, mistralBase, DoRa 7k samples, 32/64, 2e-4 lr, cosine scheduler, conversation format, response only
=== no system prompt ===
- NOT TESTING ANYMORE, UNLIKELY TO BE GOOD

=== system prompt ===
- a lot of responses are way too long, doesn't hit the stop token
- there was more failures, did not poem it up nicely
- quality in general is a drop 

# good-enough/dora-full/dora_adapter, mistralBase, DoRa 7k samples, 32/64, 2e-4 lr, cosine scheduler, conversation format
=== no system prompt ===
- NOT TESTING ANYMORE, UNLIKELY TO BE GOOD

=== system prompt ===
- half failed badly, perpetual repetition, no poem either for those
- rarely generated normal responses (not good)
- poems generated are not as high quality, dropped quality
- same goes with the latest checkpoint


# good-enough/rslora-full/dora_adapter, mistralBase, RSLoRa 7k samples, 32/64, 2e-4 lr, cosine scheduler, conversation format
=== no system prompt ===
- NOT TESTING ANYMORE, UNLIKELY TO BE GOOD

=== system prompt ===


# good-enough/dora-decay/dora_adapter, mistralBase, DoRa 1k samples, 32/64, 2e-4 lr, cosine scheduler, conversation format
=== no system prompt ===
- NOT TESTING ANYMORE, UNLIKELY TO BE GOOD

=== system prompt ===
- often repeats itself indefinitely
- when poem is generated, its good, but not great
- lengths of poems often too long, no end token made

# good-enough/dora_no_sys, base config
# good-enough/lora_no_sys, base config
- We DEFINITELY need the system prompt to get any poetry out
- DO NOT even try this again lmao

gna use DoRa, with 5% dropout, 7k full, 32/64, cosine scheduler, 5e-4 lr, conversation format, and see how it goes