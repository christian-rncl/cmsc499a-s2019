# Multitask matrix completion for protein interactions across diseases notes
### Christian Roncal, CMSC 499

# Introduction

Viruses and other pathogens infect host cells by introducing their own proteins that interact with the host cell proteins. These interactions (protein-protein interactions/PPI) enable the virus to use the host cell to replicate and evade immune system response. This paper asks the question <mark>*"Can we model host-pathogen PPIs better through data across multiple diseases?"*</mark> 


This problem can be formulated as a matrix completion problem. PPIs can be can be modelled as several bi-partite graphs over multiple hosts and pathogens. Each bipartite graph $G$ is represented through a matrix $M$. $M$'s  rows and columns represent pathogen and host proteins, respectively and $M_{ij} = 1$ for an observed interaction between pathogen protein $i$ and host protein $j$.

<center><img src='graph.png'></center>

Most prior work model pathogens separately, so there was one model for each individual bi-partite graph/pathogen-host ppi. This method can't exploit similarities between different pathogens. <mark> Here we use a multi-task approach where multiple bipartite graphs are jointly modelled, with each host/pathogen graph as a **task**. </mark> For our case, there is a single host with several different viruses. This joint representation enable us to capture similar infection strategies from similar viruses.

This model then is based on two 'biological intuitions'
1. Interactions depend on structural properties of proteins. Since most pathogens evolved from common ancestors, most of their proteins structures are similar.

2. However, different pathogens have also evolved specialized/unique mechanisms to target host proteins.

To take advantage of these two biological intuitions we write the interactions matrix $M_t$ for a task $t$ as:
$$M_t = \mu_t * \text{(shared component)} + (1 - \mu_t) * \text{specific component)}$$

## Collaborative filtering through low rank matrix factorization
