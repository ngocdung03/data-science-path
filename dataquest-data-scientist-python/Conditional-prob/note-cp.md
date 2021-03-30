##### Conditional prob: intermediate
-  Multiplication rule of probability: P(A∩B)=P(B)⋅P(A|B)
- In more general terms, if event A occurs and the probability of B remains unchanged and vice versa (A and B can be any events for any random experiment), then events A and B are said to be statistically independent (although the term "independent" is more often used).
    - P(A)=P(A|B)
    - P(A∩B)=P(A)⋅P(B)
- If multiplication events are independent, they must be: pairwise independent + mutually independent
    - Otherwise: P(A∩B∩C)=P(A)⋅P(B|A)⋅P(C | A∩B)
##### Bayes theorem
- [Prob-of-test-positive.jpg]
```
p_spam = 0.2388
p_secret_given_spam = 0.4802
p_secret_given_non_spam = 0.1284

p_non_spam = 1 - p_spam
p_spam_and_secret = p_spam*p_secret_given_spam
p_non_spam_and_secret = p_non_spam*p_secret_given_non_spam
p_secret = p_spam_and_secret + p_non_spam_and_secret
```
- *Law of total probability*: `P(A)=P(B1)⋅P(A|B1)+P(B2)⋅P(A|B2)+⋯+P(Bn)⋅P(A|Bn)`
- [Bayes-theorem0.jpg] [Bayes-theorem.jpg]
- [Relevant-formulas.jpg]