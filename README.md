# SelfIE Adapters for Auditing
This project is WIP, but here's the hypothesis & theory of change:

- [SelfIE (Self-Interpretation of Embeddings)](https://arxiv.org/html/2403.10949v2) is an approach where you ask an LLM to just explain its own activations, by plugging in the hidden states back into the model as soft input tokens. [SelfIE adapters](https://arxiv.org/html/2602.10352v2) are a way to get improved performance by applying a lightweight adapter (e.g. scale + bias) to the activation vector first.
    - This is related to the idea of Activation Oracles, where you fine-tune the model to explain its own activations. The relative advantage of SelfIE-adapter based methods is that the model's weights are frozen, so you're less likely to get confounds from the fine-tuning process itself.
    - SelfIE methods offer another potential white-box tool for interpreting models, likely with its own strengths and weaknesses, and having as diverse of a toolkit as possible helps us evaluate models for various forms of misalignment.
- I want to see if we can take the SelfIE adapter paper one step further, by training an adapter to specifically surface unverbalized concepts.
    - One of the [Jacobian lens paper](https://transformer-circuits.pub/2026/workspace/index.html#ws-modulation) experiments was to ask a model "Concentrate on X while saying Y", and finding "X" showing up in the Jacobian lens.
    - Inspired by this experiment, I want to try training an adapter, using prompts like "Think about X while saying Y", and training the SelfIE generations to recover X from the activations on the tokens Y. With some luck, the adapter will be better at surfacing hidden thoughts of the model when doing so.
- Another question is - does the adapter still work if you train it on a misaligned model (organism)?
    - This is inspired by [this paper](https://arxiv.org/pdf/2607.23379), which trained an Activation Oracle on a Taboo model organism and found the AO wasn't helpful in recovering the 
