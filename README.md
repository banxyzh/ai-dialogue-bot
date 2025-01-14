# AI Dialogue Bot with Tensorflow

This project leverages Tensorflow to build a Seq2Seq Chatbot with an attention mechanism at its core. It finds intensive use in interpreting and translating neural languages.

## Requirements
You need to have tensorflow r1.1 installed to run this project.

## References
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf)
- [A Neural Conversational Model](https://arxiv.org/pdf/1506.05869.pdf)
- [A Hierarchical Recurrent Encoder-Decoder for Generative Context-Aware Query Suggestion](https://arxiv.org/pdf/1507.02221.pdf)
- [Attention with Intention for a Neural Network Conversation Model](https://arxiv.org/pdf/1510.08565.pdf)

## Result
Here's an instance of the result achieved after training with 1400 movies and tv show subtitles (around 1.4m sentences).
![Demo](/meetup_demo.png)

## Future Work
- Use an Anti-language model to suppress generic response
- Experiment with MMI-loss as an objective function
- Consider the use of Reinforcement Learning?