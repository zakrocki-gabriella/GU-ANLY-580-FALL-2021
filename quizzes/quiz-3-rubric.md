# Quiz 3 rubric

**Format**: Canvas

- Section I: 11/15
- Section II: 11/15

**Material covered**: Lectures 07-09

#
## Topics

1. Gradient descent with NNs
    - Understand the concepts involved with performing gradient descent with NNs:
        - Chain rule
        - Derivative of a vector-valued function w.r.t. it a vector-valued input (Jacobian)
        - Backpropagation ... combining the above
        - Gradient updates ... what derivatives do we need to take in addition to above?

2. Sequence processing mechanisms with NNs
    - Have a basic understanding of the three core mechanisms for processing sequences:
        - Convolutions
        - Recurrent connections
        - Attention
        
3. Seq2seq modeling
    - How does it differ from single token classification?
    - What problems arise when trying to use a conditional language model to evaluate the probability of sequences?
        - It's computationally expensive, why?
        - Train/test distribution mismatch (i.e. the inference gap) ... what does this refer to?
        - The information bottleneck ... what does this refer to?
