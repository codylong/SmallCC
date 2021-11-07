# SmallCC

The motivation for this project is to understand whether data science techniques can aid in
finding small cosmological constants in string theory.  The two works critical to this project are:
- [Bousso-Polchinski](https://arxiv.org/abs/hep-th/0004134), which puts forth a toy flux model of the CC in IIB.
- [Douglas-Denef](https://arxiv.org/abs/hep-th/0602072), which studies the complexity of the BP model, showing it
is NP complete.

The BP model is remarkably simple and though it is a toy model it is the current gold standard. As characterized in DD,
```
L = L_0 + g_ij N_i N_j,
```
where L_0 is a bare negative real CC, N is an integral k-vector, and g_ij is a positive definite metric. 
We will let e (read: epsilon) be our standard for a small CC, i.e. a CC is small if L < e.
To my knowledge, a solution of this model consistent with the observed cosmological constant has never been
found, perhaps due to lack of sophisticated techniques, and perhaps due to complexity. A natural proposal is to
try to use reinforcement learning to find small CCs, but perhaps other techniques will provide a solution algorithm.

In what is likely the increasing order of difficulty and importance, goals for this project include:

- **Small CC:** use data science (RL?) to provide an algorithm for finding small CCs as L_0, L, and k are varied.
- **Small CC complexity:** for fixed L_0 and e, see how solution time scales with k, try different L_0 and e values. It is also important to compare this to a few different null models for the sake of comparison.
- **Observed CC:** if the former two areas show promise, then we tune paramaters and train, aiming to find a solution consistent with the observed CC.

Clearly we want to achieve all three goals, in addition to any other natural goals that arise in the project.

## Complexity 101

Since BP is NP complete, it is important to review the central types of problems in complexity theory:

- **P:** are problems in which a solution can be found in polynomial time.
- **NP:** are problems in which a proposed solution can be verified in polynomial time.
- **NP hard:** are problems to which any NP problem can be reduced. This means that a solution to an NP hard problem H can be used to produce a solution to any NP problem in polynomial time (in time units where 1 is the amount of time to solve H). NP hard problems are therefore at least as hard to solve as the hardest NP problems.
- **NP complete:** are problems that are both NP and NP hard.

The P vs. NP problem is a [million dollar problem](http://www.claymath.org/millennium-problems/p-vs-np-problem). It asks
whether or not P=NP: can any problem whose solutions may be verified in polynomial time also be solved in polynomial time? 
The consensus in the CS community is that P!=NP, though it has not been proven. By the definition of an NP hard problem,
providing a polynomial time algorithm that solves any NP hard problem would necessarily solve any NP problem, and therefore
prove P=NP.

## Approaching Bousso-Polchinski

Since the BP model is NP complete, a polynomial time solution algorithm would prove P=NP, and since it is expected
that P != NP we should not expect to find such an algorithm. 

As a basic first step, we would like to rewrite the problem a little bit. First, we write it in DD form,
```
L_1 < L_0 + g_ij N_i N_j < L_2.
```
Of course, because we are civilized, we will do everything in Planck units and set M_pl = 1.
(We will worry about
issues of computer precision later.) For simplicity we will
rewrite this as 
```
eps  < L_0 + g_ij N_i N_j < 2 eps.
```
There are four basic scales in the problem, which (biggest to smallest) are M_pl, L_0, G, eps. Here,
G is the mean value of entries of the positive definite matrix g, which are < 1 since we are in M_pl=1 units.

## Experiment Set 1 (Wishart)

We will try to make life as easy as possible to start, in order to maximize the likelihood that we can find
a solution with a hierarchically small cosmological constant. So we take G very small, and L_0 = 1. So for these
experiments, our physical input parameters are k, G, and eps.

We will take g = A^T A where the entries of A are drawn from a Gaussian centered at zero with stdev sigma, so we can
effectively replace G with sigma.
