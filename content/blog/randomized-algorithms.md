# The randomized algorithm mindset: how can we embed machine learning seamlessly into algorithmic design?

Much buzz has been 

## The toughest problems

Perhaps at the core of computer science lies the concept of algorithm: that set of instructions to be run on a Turing machine. Algorithms are solutions to all sorts of problems, from sorting numbers to efficiently selling stuff in a circuit of cities, but their general nature has not percluded their efficient classifications. When we think on algorithm classes we typically think in terms of worst case analysis, that is, given the size of the input, how many calculations does the algorithm take to finish and give us an answer in the worst possible case. This mindset pushes the algorithmic designer to generate recipes that are truly efficient no matter the input -- it is satisfying to know that sorting n numbers will always take no more than an order of nlog(n) operations. This way of thinking permeates not only algorithmic design but extends to classifying the problems these algorithms solve and gives rise to the famous P/NP classification. Unfortunately, this also puts a bit of a theoretical barrier when dealing with problems that worst-case analysis deems the hardest. Since we don't know, and it's extremely unlikely that, NP-hard problems can be solved efficiently, whenever we are confronted by one of these problems (and boy are they everywhere) we know that we will have to compromise somehow and just do our best 

## Metaheuristics

In the quest for solving these difficult problems (often combinatorial in nature), many general frameworks have beenemployed or invented. Typically, one turns to very general optimization techinques such as simulated annealing, particle swarm optimization, cross entropy, tabu search, genetic algorithms, and, more recently, even deep reinforcement learning. These are sometimes called metaheuristics, since they are not algorithms per se, but rather recipes to create algorithms that can solve a particular task. In essence, all of them present ways of somehow searching or sampling the massive solution space efficiently to find a reasonable solution. If you're tackling the travelling salesman problem, then it becomes the space of paths within a graph. How efficiently the space is searched will depend on the nature (and sometimes instance!) of the problem and of the method you're trying: e.g. for some types of graphs, the traveling salesman problem will be better approximated with say tabu search while in others reinforcement learning will work pretty well. In other words, your algorithm will be tailored to the data you are analyzing. It might not be general enough to solve all instances of the problem, but for practical purposes it doesn't need to be. Unfortunately, this also makes the general analysis of algorithms produced by a metaheuristic more complicated and it's hard to get good theoretical guarantees. An additional complexity is that the effectiveness applying a metaheuristic will also depend on how familiarized one is with the metaheuristic. Practitioners of each metaheuristic typically have rules of thumbs (heuristics for metaheuristics!) for generating a good approximation algorithms given each situation. 

## Randomized Algorithms

There is another deceptively simple way of approximating NP-hard problems: just by sheer luck. Randomized algorithms are algorithms in which one step or variable is treated as a random variable. The most famous randomized algorithm is probably the list-sorting algorithm quicksort. Just randomly choose a random pivot on the list, divide, and repeat. The reason it's so popular even though it is O(n^2) is that it is so easy to understand and to implement. Who cares that the worst case is bad, when analyzing the average case among lists of comparable items, quicksort is still just nlog(n). That's the gist of randomized algorithms. They are typically very simple and clever, easy to understand and implement, but play a game of luck for the expected quality of the solution -- when analyzing them, you typically see statements that they approximate the optimal solution by a factor of X "with high probability". 

Note that most (all?) of the algorithms produced through metaheuristics are in fact randomized algorithms by definition since they have to sample the solution space somehow. However, at least to my appreciation, the algorithms produced by the randomized algorithms field are different in spirit: they are constructs that try to be more general and give good theoretical guarantees (or in some cases tight approximation factor bounds). They are simple and well understood, but their solutions are typically inferior to well tuned metaheuristic instances that are massively sampled. Keeping this in mind, I will refer to randomized algorithms as algorithms coming from that field, which is something separate to their metaheuristic kin.

## Learnlets: smart de-randomization using embedded ML

What I love about randomized algorithms is how straightforward they are to implement. What I love about metaheuristics is their flexibility that allows them to be tuned to my particular dataset. Is there a way to bring the two together? I think a good answer lies in the old adage: "implement now, optimize later". My ideal workflow would go like this: 

1. For a given NP-hard problem you code up a super simple randomized algorithm that produces a (very crude) approximate solution by taking random decisions at critical points where some heavy, exponential computation would otherwise be required.
2. You somehow flag those random, critical decision points to be optimized later.
3. You go about your way using the algorithm, getting a sense of what data is it most commonly going to be used on.
4. At some point, after collecting or generating data that your algorithm is being applied on, the system optimizes the flagged critical decision points, tailoring it to your data, using a metaheuristic. The random decisions points are gone, you have effectively de-randomized your algorithm.
5. The system continuously learns as you keep calling your algorithm with new data, optimizing the decision points periodically as new data comes in.

In the end, you would have coded up a 'dumb' algorithm that gets better (for your particular task at least) as you keep using it. For lack of a better term, I'll be calling these trainable randomized algorithms *learnlets*, pieces of code with a bit of embedded machine learning that learn as they go, and the critical decision points that will be learned/optimized as *learnables*.

## A python example: the MAXCUT problem

To ground the concept, let's look at an example. Consider the MAXCUT problem: given a weighted graph, find an optimal way to split it into two subgraphs such that the sum of edge weights between subgraphs is maximum. MAXCUT is NP-hard and is typically used as one of the first examples for illustrating randomized algorithms. The naive randomized algorithm for MAXCUT is extremely simple: for each vertex, randomly choose if it will be part of one subgraph or the other. A simple analysis reveals that this algorithm gives an expected cut of at least half the amount of the actual maximum cut, so in a way it gets us halfway there. This is the python implementation:


Notice how I'm using the graph vertex degrees as input to the random decision function `vertex_set_probs`, even though I don't use them at all. This is intentional, I'm thinking ahead in that these will be the features that will be used to better the improve the algorithm. It also outputs probabilities of sets, which will be useful for scoring solutions, we can at anytime threshold these proabilities at 0.5 to decide which vertex belongs to which set. Now, what I want to do is somehow turn this into a learnlet, by flagging the `vertex_set_probs` function as a learnable decision function. To this end, I've implemented a very simple learnlet framework in python that uses a crude reinforcement learning strategy using policy gradients to optimize the learnables given some score function, leveraging the excellent autograd package. My framework uses decorators to flag functions as learnets or learnables, which is a nice way to seamlessly add functionality to your code:



Some explanations of the arguments of the decorators are in order. The learnable decorator specifies that the function will be approximated with a particular predictor, in this case a very simple dense neural net. The learnlet decorator simply flags the function as a learnlet and lists all of the learnables that it uses. Notice that the first (input) layer of the `vertex_set_probs` predictor neural net is set to 20. Since the function takes an array of vertex degrees, this means that our current learnlet will only work with graphs with 20 vertices. 

Now, let's generate some data. Suppose that this code will mostly run on graphs of 20 vertices that have two cliques chosen at random whith sparse, inter-clique connections. Further, suppose that the cliques have weak intra-clique connections but very strong inter-clique connections, so the max cut is obvious. Here's the code for generating these graphs and an example adjacency matrix.


I'll generate some training and testing graphs for our learnlet.

Next, I have to write a function that scores my solutions, so here it is.


So the score function is basically the expected value of the cut given the membership probabilities. Why didn't I use the more straightforward score that thresholds the probabilities and simply takes the value of the resulting cut? Turns out that this results in a difficult-to-optimize function that varies step-wise (small changes in membership probability will yield the same cut) and so when doing gradient descent you end up with very flat regions of no change that mess up the calculations.
