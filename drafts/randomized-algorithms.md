# Learnlets: an attempt to seamlessly embed machine learning into algorithmic design

There has been a lot of excitement lately on using machine learning to reinvent the way we write software, even to the point where one of the godfathers of deep learning wants to rebrand deep learning as differential programming. Indeed, why tell the computer exactly what to do? Just give it examples of what you want to do, set up a model, and let machine learning figure out the rest. For many human-like tasks where we have a decent ammount of data already collected, such as question and answering or summarizing text, this is the currently our best approach since designing a universal set of instructions by hand that can solve such variable tasks in every context is likely an impossible task. But there are other problems that admit very general and efficient algorithmic solutions that would, at first glance, seem silly to code with machine learning. Why train a neural network to sort numbers, or to find the maximum flow in the graph, or to solve a linear program where a real neural network (that of a human) already codified it into a pretty general recipe that works well and fast? In this post, I explore whether even those tasks can benefit from machine learning models, that is, whether there are benefits to embedding machine learning models into algorithmic design and implementation. The key idea is to consider that any algorithm that we implement will be eventually be run on a very small and likely correlated space of all possible inputs. Thus, tailoring algorithmic implementations to the data that they will be used on might result in performance improvement -- e.g. even a simple sorting algorithm can benefit knowing, beforehand, that half of the list is already sorted. This way of conceptualizing the coding of algorithms leads to all sorts of interesting consequences: coding primarily as a means to get data, problem complexity as a function of data, compiling the algorithm structure itself, and the need to embed data into package management. This viewpoint is best illustrated when dealing with NP-hard problems and is exemplified neatly by one strategy to tackle these tough problems: randomized algorithms. To ground these ideas, I'll sketch a coding framework, called learnlets, for embedding ML into algorithmic design and a python proof-of-concept implementation that may be one of many ways of getting to software 2.0. Full disclosure: I'm no expert in automatic program synthesis or related fields, this is just a set of ideas that I wanted to write down -- let me know if I'm missing any related work.

## The toughest problems

Algorithms are at the core of computer science and are solutions to all sorts of problems, from sorting numbers to efficiently selling stuff in a circuit of cities. When classifying algorihtms, we typically think in terms of worst case analysis, that is, given the size of the input, how many calculations does the algorithm take to finish and give us an answer in the worst possible case. This mindset is a high bar that pushes the algorithmic designer to generate recipes that are truly efficient no matter the input and come with a guaranteed running time -- it is satisfying to know that sorting n numbers will always take no more than an order of *nlog(n)* operations. This way of thinking permeates not only algorithmic design but extends to classifying the problems these algorithms solve and gives rise to the famous P/NP classification. Unfortunately, this also puts a bit of a psychological  barrier when dealing with problems that worst-case analysis deems the hardest. Since we don't know, and it's extremely unlikely that, NP-hard problems can be solved efficiently in general, whenever we are confronted by one of these problems (and boy are they everywhere) we know that we will have to compromise somehow and just do our best to find an approximate solution. 

## Metaheuristics

So how does one go about to solve these hard problems? One option is to turn to very general optimization techinques such as simulated annealing, particle swarm optimization, cross entropy, tabu search, genetic algorithms, and even deep reinforcement learning. These are sometimes called metaheuristics, since they are not algorithms per se, but rather recipes to create algorithms that can solve a particular task. In essence, all of them present ways of somehow searching or sampling the massive solution space efficiently to find a reasonable solution. If you're tackling the travelling salesman problem, then it becomes the space of paths within a graph. How efficiently the space is searched will depend on the nature (and sometimes instance!) of the problem and of the method you're trying: e.g. for some types of graphs, the traveling salesman problem will be better approximated with say tabu search while in others reinforcement learning will work pretty well. In other words, your algorithm will be tailored to the particular instance you are analyzing. It might not be general enough to solve all instances of the problem, but for practical purposes it doesn't need to be. Unfortunately, this also makes the general analysis of algorithms produced by a metaheuristic more complicated and it's hard to get good theoretical guarantees. An additional complexity is that the effectiveness applying a metaheuristic will also depend on how familiarized one is with the metaheuristic and on intuitions on how to obtain good representations of the input data. Practitioners of each metaheuristic typically have rules of thumbs (heuristics for metaheuristics!) for generating a good approximation algorithms given each situation. 

## Randomized Algorithms

There is another deceptively simple way of approximating NP-hard problems: just by sheer luck. Randomized algorithms are algorithms in which one step or variable is treated as a random variable. The most famous randomized algorithm is probably the list-sorting algorithm quicksort. Just randomly choose a random pivot on the list, divide, and repeat. The reason it's so popular even though it is O(n^2) is that it is so easy to understand and to implement. Who cares that the worst case is bad, when analyzing the average case among lists of comparable items, quicksort is still just nlog(n). That's the gist of randomized algorithms. They are typically very simple and clever, easy to understand and implement, but play a game of luck for the expected quality of the solution -- when analyzing them, you typically see statements that they approximate the optimal solution by a factor of X "with high probability". Other flavors of randomized algorithms build on well known optimization techniques like linear or semidefinite programming, relaxing what's typically a combinatorial problem of picking integer solution to a continuous optimization problem of picking probabilities.

Note that most (all?) of the algorithms produced through metaheuristics are in fact randomized algorithms by definition since they have to sample the solution space somehow. However, at least to my appreciation, the algorithms produced by the randomized algorithms field are different in spirit: they are constructs that try to be more general and give good theoretical guarantees (or in some cases tight approximation factor bounds). They are simple and well understood, but their solutions are typically inferior to well-tuned metaheuristic instances that are massively sampled or . Keeping this in mind, I will refer to randomized algorithms as algorithms coming from that field, which is something separate to their metaheuristic kin.

## Learnlets: smart de-randomization using embedded ML

What I love about randomized algorithms is how straightforward they are to implement. What I love about metaheuristics is their flexibility that allows them to be tuned to my particular dataset. Is there a way to bring the two together? I think a good answer lies in the old adage: "implement now, optimize later". My ideal workflow would go like this: 

1. For a given NP-hard problem you code up a super simple randomized algorithm that produces a (very crude) approximate solution by taking random decisions at critical points where some heavy, exponential computation would otherwise be required.
2. You somehow flag those random, critical decision points to be optimized later.
3. You go about your way using the algorithm, getting a sense of what data is it most commonly going to be used on.
4. At some point, after collecting or generating data that your algorithm is being applied on, the system optimizes the flagged critical decision points, tailoring it to your data, using a metaheuristic. The random decisions points are gone, you have effectively de-randomized your algorithm.
5. The system continuously learns as you keep calling your algorithm with new data, optimizing the decision points periodically as new data comes in.

In the end, you would have coded up a 'dumb' algorithm that gets better (for your particular task at least) as you keep using it. I'll be calling these trainable randomized algorithms *learnlets*, pieces of code with a bit of embedded machine learning that learn as they go, and the critical decision points that will be learned/optimized as *learnables*.

## A python example: the MAXCUT problem

To ground the concept, let's look at an example. Consider the [MAXCUT problem](https://en.wikipedia.org/wiki/Maximum_cut): given a weighted graph, find an optimal way to split it into two subgraphs such that the sum of edge weights between subgraphs is maximum. MAXCUT is NP-hard and is typically used as one of the [first examples](http://www.cs.cornell.edu/courses/cs4820/2011sp/handouts/approx_algs.pdf) for illustrating randomized approximation algorithms. The naive randomized algorithm for MAXCUT is extremely simple: for each vertex, randomly choose if it will be part of one subgraph or the other. A simple analysis reveals that this algorithm gives an expected cut of at least half the amount of the actual maximum cut, so in a way it gets us halfway there. This is the python implementation:

{{< highlight python >}}
def vertex_set_probs(adj_mat_flattened):
    return np.random.rand(GRAPH_SIZE)

def rand_max_cut(adj_mat):
    chosen_probs = vertex_set_probs(adj_mat.ravel())
    return adj_mat, chosen_probs.reshape([chosen_probs.shape[0], 1])

{{< /highlight >}}

Notice how I'm using the graph's adjacency matrix as input to the random decision function `vertex_set_probs`, even though I don't use itat all. This is intentional, I'm thinking ahead in that these will be the features that will be used to improve the algorithm. It also outputs probabilities of sets, which will be useful for scoring solutions, we can at anytime threshold these proabilities at 0.5 to decide which vertex belongs to which set. Now, what I want to do is somehow turn this into a learnlet, by flagging the `vertex_set_probs` function as a learnable decision function. To this end, I've implemented a very simple learnlet framework in python that uses a crude reinforcement learning strategy using policy gradients to optimize the learnables given some score function, leveraging the excellent [autograd package](https://github.com/HIPS/autograd). Inspired by the awesome [numba module](https://numba.pydata.org/), my framework uses decorators to flag functions as learnets or learnables, which is a nice way to seamlessly add functionality to your code (you can find the whole implementation and the examples of this post in a jupyter notebook here):

{{< highlight python >}}
@learnable(predictor=DenseNetPredictor(layer_sizes=[20*20, 20, 20],
                                       activations=[relu, sigmoid]))
def vertex_set_probs(adj_mat_flattened):
    return np.random.rand(GRAPH_SIZE)

@learnlet(learnables=[vertex_set_probs])
def rand_max_cut(adj_mat):
    degrees = adj_mat.sum(axis=1)
    chosen_probs = vertex_set_probs(adj_mat.ravel())
    return adj_mat, chosen_probs.reshape([chosen_probs.shape[0], 1])

{{< /highlight >}}

Some explanations of the arguments of the decorators are in order. The learnable decorator specifies that the function will be approximated with a particular predictor, in this case a very simple dense neural net. The learnlet decorator simply flags the function as a learnlet and lists all of the learnables that it uses. Notice that the first (input) layer of the `vertex_set_probs` predictor neural net is set to 20. Since the function takes an array of vertex degrees, this means that our current learnlet will only work with graphs with 20 vertices. 

Now, let's generate some data. Suppose that this code will mostly run on graphs of 20 vertices that have two cliques chosen at random whith sparse, inter-clique connections. Further, suppose that the cliques have weak intra-clique connections but very strong inter-clique connections, so the max cut is obvious. Here's the code for generating these graphs and an example adjacency matrix.

{{< highlight python >}}
from itertools import product

GRAPH_SIZE = 20

def submat_indices(idxset1, idxset2):
    I = np.array([x for x in product(idxset1, idxset2)])
    return I[:, 0], I[:, 1]

def generate_graph(size):
    set1 = np.random.choice(range(size), np.random.randint(5, np.int(size/2)), replace=False)
    set2 = np.array([x for x in range(size) if x not in set1])
    
    adj_mat = np.eye(size)
    adj_mat[submat_indices(set1, set1)] = 3
    adj_mat[submat_indices(set2, set2)] = 1
    
    interface_nodes_set1 = np.random.choice(set1, 5, replace=False)
    interface_nodes_set2 = np.random.choice(set2, 5, replace=False)
    
    
    adj_mat[submat_indices(interface_nodes_set1, interface_nodes_set2)] = 50
    adj_mat[submat_indices(interface_nodes_set2, interface_nodes_set1)] = 50
    
    r = np.arange(size)
    r1 = np.zeros([size, 1])
    r1[np.isin(r, set1)] = 1
    
    adj_mat += np.random.rand(*adj_mat.shape)
    adj_mat = (adj_mat + adj_mat.T)/2.
    return adj_mat, r1
{{< /highlight >}}


I'll generate some training and testing graphs for our learnlet.

{{< highlight python >}}
train_graphs_and_truth = [generate_graph(size=GRAPH_SIZE) for _ in range(100)]
test_graphs_and_truth = [generate_graph(size=GRAPH_SIZE) for _ in range(100)]

train_graphs = [x[0] for x in train_graphs_and_truth]
test_graphs = [x[0] for x in test_graphs_and_truth]
{{< /highlight >}}

Next, I have to write a function that scores my solutions, so here it is.

{{< highlight python >}}
def max_cut_score(max_cut_results):
    scores = []
    for adj_mat, chosen_probs in max_cut_results:
        p = np.matmul(1 - chosen_probs, chosen_probs.T)
        scores.append((p * adj_mat).sum()/2.)
    return np.array(scores).mean()
{{< /highlight >}}

So the score function is basically the expected value of the cut given the membership probabilities. Why didn't I use the more straightforward score that thresholds the probabilities and simply takes the value of the resulting cut? Turns out that this results in a difficult-to-optimize function that varies step-wise (small changes in membership probability will yield the same cut) and so when doing gradient descent you end up with very flat regions of no change that mess up the calculations.

Alright, we're ready to optimize our learnlet!

{{< highlight python >}}
optimize(rand_max_cut, 
         score_fun=max_cut_score, 
         train_args=dict(step_size=0.1, num_iters=100),
         batch_size=50,
         l2_reg=0.6,
         data=train_graphs)
{{< /highlight >}}

The `optimize` function takes typical optimization hyperparameters to perform gradient descent on neural nets and then optimizes the predictors for each learnable in the specified learnlet. Let's run our learnlet with and without using the predictors in the train/test data.

{{< highlight python >}}
from collections import OrderedDict
import pandas as pd

def run_on_graphs(graphs, predict=False):
    results = []
    for adj_mat in graphs:
        _, chosen_probs = rand_max_cut(adj_mat, predict=predict)
        results.append((adj_mat, chosen_probs))
    return results


random_train_results = run_on_graphs(train_graphs, predict=False)
random_test_results = run_on_graphs(test_graphs, predict=False)

train_results = run_on_graphs(train_graphs, predict=True)
test_results = run_on_graphs(test_graphs, predict=True)
{{< /highlight >}}

Notice the `predict=True` argument when running the max cut function. This is an extra keyword argument that is added by the learnlet decorator that switches the learnlet to prediction mode, in which it uses the learnable predictors instead of the hard-coded, random function to execute the algorithm. Let's now compare the results of our random baseline with our de-randomized learnlet as well to the "ground truth".

{{< highlight python >}}
results_dict = OrderedDict()
results_dict['test'] = dict(random=max_cut_score(random_test_results),
                            learnlet=max_cut_score(test_results),
                            truth=max_cut_score(test_graphs_and_truth))
results_dict['train'] = dict(random=max_cut_score(random_train_results), 
                             learnlet=max_cut_score(train_results),
                             truth=max_cut_score(train_graphs_and_truth))


pd.DataFrame.from_dict(results_dict, orient='index')
{{< /highlight >}}

Cool! Something definitely happened, the score is up in both the train and test sets, so our learnlet did learn something. It didn't quite pick up the optimal value which is a shame, although that would be difficult using such an unstructured representation of the adjacency matrix. Still, we basically got this performance boost for free, just adding a couple of decorators to our functions. We can confirm that our learned solutions are better than baseline using the standard max cut score instead of the expected score:

{{< highlight python >}}
def max_cut_rounding_score(max_cut_results):
    scores = []
    for adj_mat, chosen_probs in max_cut_results:
        set1 = np.where(chosen_probs >= 0.5)[0]
        set2 = np.where(chosen_probs < 0.5)[0]
        scores.append(adj_mat[submat_indices(set1, set2)].sum())
    return np.array(scores).mean()
    

results_dict = OrderedDict()
results_dict['test'] = dict(random=max_cut_rounding_score(random_test_results),
                            learnlet=max_cut_rounding_score(test_results),
                            truth=max_cut_rounding_score(test_graphs_and_truth))
results_dict['train'] = dict(random=max_cut_rounding_score(random_train_results), 
                             learnlet=max_cut_rounding_score(train_results),
                             truth=max_cut_rounding_score(train_graphs_and_truth))



pd.DataFrame.from_dict(results_dict, orient='index')
{{< /highlight >}}

### Comparing to a more sophisticated randomized algorithm

A random baseline is tough not to beat, what if we up the ante and turn to a better randomized algorithm? There is a well-known MAXCUT randomized algorithm based on randomized semidefinite programming. The main idea is that you can cast MAXCUT as an integer programming problem, solve a continuous relaxation of it, and use the continuous solution to map to a decision vector that puts each node in one of the two subgraphs that define the cut. Using the randsdp library, this is what the problem looks like in python:

{{< highlight python >}}
import randsdp as rsdp
import cvxpy as cvx

def semidef_max_cut(adj_mat):
    n = adj_mat.shape[0]
    X = rsdp.Semidef(n)
    constraints = [cvx.diag(X) == 1.0]
    f = rsdp.Trace(X, C=adj_mat)
    problem = rsdp.Problem(f, constraints)
    problem.solve(verbose=False)
    probs = (np.dot(X.value, np.random.randn(n, 100)) > 0).mean(axis=1)
    return np.array(probs)
{{< /highlight >}}

So what happens when we run this on our graphs? In expectation we actually get a worse solution than our learnlet

But when doing the rounding, we are blown out of the water!



The current learnlet approach still needs improving in several areas. For example, the hyperparameters I chose for the neural net predictor and the optimizer required a bit of experimentation and is something that could be totally automated away using auto ML techniques. Other optimization metaheuristics could be implemented as well, and parallelization of the learning routines, as well as using more scalable machine learning frameworks like PyTorch, Tensorflow, with libraries like Keras to more easily search the space of possible architectures would be desirable. 

### Recording the training data

I've also experimented with adding ways for automatically capturing the training data of a learnlet: e.g. each time a learnlet-decorated function is called, save the arguments in some database and then use the aggregated data to train the learnlet. For the testing purposes above, I found that this was getting in the way as I needed to measure performance with test/train sets. However, in the real world and when using the algorithm, one could see that a data-capturing feature would be necessary in order to understand the data context where the code is being used. This is an example of my current solution to this problem:

{{< highlight python >}}
for adj_mat in train_graphs:
    # The record flag saves the data (adj_mat) in a database (in this case a simple dict) to be used for training later
    rand_max_cut(adj_mat, record=True)

# Notice the lack of data argument, the recorded data is used to train the learnlet
optimize(rand_max_cut, 
         score_fun=max_cut_score, 
         train_args=dict(step_size=0.1, num_iters=100),
         batch_size=50,
         l2_reg=0.6)
{{< /highlight >}}

## Implications of embedding ML into algorithmic design

What are the implications of this way of coding algorithms? Here are some that I think are interesting. 

### Algorithm compiling

For decades, we've used compilers to optimize the code we write, translating it to efficient machine code that rearranges our loops, conditionals, etc. so that it runs faster than simply transcribing our instructions as-is to op codes. The instructions are changed by the compiler, but the essence of the algorithm remains the same. But how about "compiling" the structure of the algorithm itself, optimizing not only its speed but the quality of its solutions given our data? I think this is a task that is well-suited for a combination of machine learning with traditional programming since we can seamlessly combine the two to get better results with little effort, like in the learnlet framework described above. In a way, in this "algorithm compiling" workflow, one uses an easily-implemented but dumbed-down algorithm as an instrument to quickly capture the data context of the problem and that serves as minimal control logic, where the heavy stuff is handed down to the embedded ML.

### P vs NP, whithin the context of your data

In hindsight, it seems surprising that it's not standard practice to record (or at least record periodically) the data each function in our codebase is being called on as it pretty much defines the worlds our programs live in. It also defines how efficient and effective they are going to be and even what unit tests should be written. No algorithm is going to be run on all the possible inputs and in most cases the worst-case input simply doesn't happen. Consequently, and loosely speaking, the definitions of P/NP change between universes of data. In some data contexts, the travelling salesman problem could be optimally approximated by a greedy algorithm. Learning these best approximations automatically within the data at hand is where ML could really shines.

### Package management, with data

Say you coded up a learnlet and trained it on your data. Say your data is general enough that some other folks might also have similar data and could use the same optimized learnlet. How do you share your trained learnlet with them? Is it part of your package when you deploy in your favority package management system? But what if your data is private? These are tough questions, but we may already have the technology to answer them. There could be repositories of learned models that you could run your data against, securely, to see if it matches, and then get an already optimized learnlet as your starting point. Performing these checks securely is something than could be done in e.g. a blockchain, similarly as how openmined trains models in a trustless, distributed setting -- so package management, put it on the blockchain!

## Challenges and opportunities

Using ML as a main driver when coding presents both opportunities and challenges. Some steps required to get good models and therefore good programs are still inefficient. For example, performing Auto ML to find the best model is still very time-consuming and many times resource hungry, so if you want to 'compile your algorithm' like mentioned above you might have to do it in the cloud and you better have a credit card ready. Debugging could be a nightmare: how do you know that your trained model is failing you because of the data itself or because of some missing data preprocessing step before you trained your model?  Recording all the data that your program uses might be bloated, undesirable, you'll likely have to make a decision which data streams are captured. In this case, thinking of compression and coding strategies to just save the data that best describes your data distribution could be helpful. 

These are only ideas in a brainstorm, implementing and testing them will take time, expertise, and diligence. But the investment will be worth it, in the end, the whole practice of programming will be reduced to writing data collection and control logic for running smart modules that learn to perform well within your data universe, just as become herders of messages in classic web apps, sometimes forgetting the complex core that really does the bulk of the computation.
