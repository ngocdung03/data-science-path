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
- The probability of being infected with HIV before doing any test is called *the prior probability* ("prior" means "before"). The probability of being infected with HIV after testing positive is called *the posterior probability* ("posterior" means "after"). 
```py
p_spam = 0.2388
p_secret_given_spam = 0.4802
p_secret_given_non_spam = 0.1284

p_spam_given_secret = (p_spam*p_secret_given_spam)/(p_secret_given_spam*p_spam+p_secret_given_non_spam*(1-p_spam))
prior = p_spam
posterior = p_spam_given_secret
ratio = posterior/prior
```
##### The Naive Bayes Algorithm
- Ignore the division - P(New message). 
```py
p_spam = 0.5
p_non_spam = 0.5
p_new_message_given_spam = 0.75
p_new_message_given_non_spam = 0.3334

p_spam_given_new_message = p_spam*p_new_message_given_spam
p_non_spam_given_new_message = p_non_spam*p_new_message_given_non_spam
ratio = p_spam_given_new_message/p_non_spam_given_new_message
classification = 'spam'
```
- Whenever we have to deal with words that are not part of the vocabulary, one solution is to ignore them when we're calculating probabilities. If we wanted to calculate P(Spam|"secret code to unlock the money"), we could skip calculating P("code"|Spam), P("to"|Spam), and P("unlock"|Spam) because "code", "to", and "unlock" are not part of the vocabulary.
- P(Spam|"secret code to unlock the money") and P(SpamC|"secret code to unlock the money") were equal to 0. This will always happen when we have words that occur in only one category.
    - Additive smoothing for every word (both exclusive and non): [Additive-smoothing.jpg]
    - Laplace smoothing/add-one smoothing, Lidstone smoothing
- Multinomial Naive Bayes, Gaussian Naive Bayes, Bernoulli Naive Bayes