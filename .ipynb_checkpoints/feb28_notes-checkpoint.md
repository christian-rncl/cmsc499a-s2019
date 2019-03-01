# February 28 Notes

Got back to implementing, starting with the data class. I want to experiment with different models so I wanted 
it to have a more modular design. 

## Dimensionality mismatch and hack
Found that the virus and human datasets have different dimensionalities. $\hat{y} \in \mathbb{R}^{(1107, 2731)}$ and $\hat{x} \in \mathbb{R}^{(7209, 2793)}$ In the paper both should have a dim2 = 2793. The hacks seems to be that there is an extra value in all feature vectors: 2799:1 which makes both have dim2 = 2799. I've noted this since I could just be misunderstanding and/or there might be a purpose for this (bias term????). I'm also not aware of its implications.