# Feb 28 - Mar 01 update
## Christian Roncal, CMSC499A Leiserson

### February 28 - Mar 01

### **Summary:**
<p>I found this paper <a href='https://arxiv.org/abs/1708.05031'>"Neural Collaborative Filtering"</a> and a 
    <a href='https://github.com/LaceyChen17/neural-collaborative-filtering'> pytorch implementation </a>
in pytorch. The implementation and the part about "general matrix factorization"
    made me understand matrix factorization for collaborative filtering a lot better. I just ran with the hype
and learned mostly deep learning for CV with little traditional ML. The deep learning pov for
this problem, helped a lot. I've began work on a framework that will let me do experiments easily and have made progress testing it with single task matrix factorization.</p>

### **Progress:**
<p>
    I've started building a framework based on the implementation I found that will let me test for 
    single/multiple species, and experiment with different collaborative filtering models. 
</p>

<p> I'm finishing up writing Matrix factorization model for single task learning. I'm now finishing up evaluation and k folds cross validation. Training is working well so far, being able to overfit. I also figured out how to use tensorboard with pytorch.</p>

### **Questions/Concerns:**
#### Dimensionality mismatch and hack
<p>
Found that the virus and human datasets have different dimensionalities. $\hat{y} \in \mathbb{R}^{(1107, 2731)}$ and $\hat{x} \in \mathbb{R}^{(7209, 2793)}$ In the paper both should have a dim2 = 2793. The hacks seems to be that there is an extra value in all feature vectors: 2799:1 which makes both have dim2 = 2799. I've noted this since I could just be misunderstanding and/or there might be a purpose for this (bias term????). I'm also not aware of its implications.
 </p>
 