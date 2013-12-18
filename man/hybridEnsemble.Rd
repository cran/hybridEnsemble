\name{hybridEnsemble}
\alias{hybridEnsemble}
\title{Binary classification with Hybrid Ensemble}
\usage{
  hybridEnsemble(x = NULL, y = NULL, combine = NULL,
    eval.measure = "auc", verbose = FALSE, RF.ntree = 500,
    AB.iter = 500, AB.maxdepth = 3, KF.cp = 1,
    KF.rp = round(log(ncol(x) + 1, 4)), NN.rang = 0.1,
    NN.maxit = 10000, NN.size = c(5, 10, 20),
    NN.decay = c(0, 0.001, 0.01, 0.1),
    SV.gamma = 2^(-15:3), SV.cost = 2^(-5:13),
    SV.degree = c(2, 3),
    SV.kernel = c("radial", "sigmoid", "linear", "polynomial"),
    rbga.popSize = 42, rbga.iters = 300,
    rbga.mutationChance = 1/rbga.popSize,
    rbga.elitism = max(1, round(rbga.popSize * 0.05)),
    DEopt.nP = 20, DEopt.nG = 300, DEopt.F = 0.9314,
    DEopt.CR = 0.6938, GenSA.maxit = 300,
    GenSA.temperature = 0.5, GenSA.visiting.param = 2.7,
    GenSA.acceptance.param = -5, GenSA.max.call = 1e+07,
    malschains.popsize = 60, malschains.ls = "cmaes",
    malschains.istep = 300, malschains.effort = 0.5,
    malschains.alpha = 0.5, malschains.threshold = 1e-08,
    malschains.maxEvals = 300, psoptim.maxit = 300,
    psoptim.maxf = Inf, psoptim.abstol = -Inf,
    psoptim.reltol = 0, psoptim.s = 40, psoptim.k = 3,
    psoptim.p = 1 - (1 - 1/psoptim.s)^psoptim.k,
    psoptim.w = 1/(2 * log(2)), psoptim.c.p = 0.5 + log(2),
    psoptim.c.g = 0.5 + log(2), soma.pathLength = 3,
    soma.stepLength = 0.11, soma.perturbationChance = 0.1,
    soma.minAbsoluteSep = 0, soma.minRelativeSep = 0.001,
    soma.nMigrations = 300, soma.populationSize = 10,
    tabu.iters = 300, tabu.listSize = c(5:12))
}
\arguments{
  \item{x}{A data frame of predictors. Categorical
  variables need to be transformed to binary (dummy)
  factors.}

  \item{y}{A factor of observed class labels (responses)
  with the only allowed values \{0,1\}.,}

  \item{combine}{Additional methods for combining the
  sub-ensembles. The simple mean, authority-based weighting
  and the single best are automatically provided since they
  are very effficient.  Possible additional methods:
  Genetic Algorithm: "rbga", Differential Evolutionary
  Algorithm: "DEopt", Generalized Simulated Annealing:
  "GenSA", Memetic Algorithm with Local Search Chains:
  "malschains", Particle Swarm Optimization: "psoptim",
  Self-Organising Migrating Algorithm: "soma", Tabu Search
  Algorithm: "tabu", Non-negative binomial likelihood:
  "NNloglik", Goldfarb-Idnani Non-negative least squares:
  "GINNLS", Lawson-Hanson Non-negative least squares:
  "LHNNLS".}

  \item{eval.measure}{Evaluation measure for the following
  combination methods: authority-based method, single best,
  "rbga",
  "DEopt","GenSA","malschains","psoptim","soma","tabu".
  Default is the area under the receiver operator
  characteristic curve 'auc'. The area under the
  sensitivity curve ('sens') and the area under the
  specificity curve ('spec') are also supported.}

  \item{verbose}{TRUE or FALSE. Should information be
  printed to the screen while estimating the Hybrid
  Ensemble.}

  \item{RF.ntree}{Random Forest parameter. Number of trees
  to grow.}

  \item{AB.iter}{Stochastic AdaBoost parameter. Number of
  boosting iterations to perform.}

  \item{AB.maxdepth}{Stochastic AdaBoost parameter. The
  maximum depth of any node of the final tree, with the
  root node counted as depth 0.}

  \item{KF.cp}{Kernel Factory parameter. The number of
  column partitions.}

  \item{KF.rp}{Kernel Factory parameter. The number of row
  partitions.}

  \item{NN.rang}{Neural Network parameter. Initial random
  weights on [-rang, rang].}

  \item{NN.maxit}{Neural Network parameter. Maximum number
  of iterations.}

  \item{NN.size}{Neural Network parameter. Number of units
  in the single hidden layer.}

  \item{NN.decay}{Neural Network parameter. Weight decay.}

  \item{SV.gamma}{Support Vector Machines parameter. Width
  of the Guassian for radial basis and sigmoid kernel.}

  \item{SV.cost}{Support Vector Machines parameter. Penalty
  (soft margin constant).}

  \item{SV.degree}{Support Vector Machines parameter.
  Degree of the polynomial kernel.}

  \item{SV.kernel}{Support Vector Machines parameter.
  Kernels to try. Can be one or more of:
  'radial','sigmoid','linear','polynomial'.}

  \item{rbga.popSize}{Genetic Algorithm parameter.
  Population size.}

  \item{rbga.iters}{Genetic Algorithm parameter.  Number of
  iterations.}

  \item{rbga.mutationChance}{Genetic Algorithm parameter.
  The chance that a gene in the chromosome mutates.}

  \item{rbga.elitism}{Genetic Algorithm parameter. Number
  of chromosomes that are kept into the next generation.}

  \item{DEopt.nP}{Differential Evolutionary Algorithm
  parameter. Population size.}

  \item{DEopt.nG}{Differential Evolutionary Algorithm
  parameter. Number of generations.}

  \item{DEopt.F}{Differential Evolutionary Algorithm
  parameter. Step size.}

  \item{DEopt.CR}{Differential Evolutionary Algorithm
  parameter. Probability of crossover.}

  \item{GenSA.maxit}{Generalized Simulated Annealing.
  Maximum number of iterations.}

  \item{GenSA.temperature}{Generalized Simulated Annealing.
  Initial value for temperature.}

  \item{GenSA.visiting.param}{Generalized Simulated
  Annealing. Parameter for visiting distribution.}

  \item{GenSA.acceptance.param}{Generalized Simulated
  Annealing. Parameter for acceptance distribution.}

  \item{GenSA.max.call}{Generalized Simulated Annealing.
  Maximum number of calls of the objective function.}

  \item{malschains.popsize}{Memetic Algorithm with Local
  Search Chains parameter. Population size.}

  \item{malschains.ls}{Memetic Algorithm with Local Search
  Chains parameter. Local search method.}

  \item{malschains.istep}{Memetic Algorithm with Local
  Search Chains parameter. Number of iterations of the
  local search.}

  \item{malschains.effort}{Memetic Algorithm with Local
  Search Chains parameter. Value between 0 and 1. The ratio
  between the number of evaluations for the local search
  and for the evolutionary algorithm. A higher effort means
  more evaluations for the evolutionary algorithm.}

  \item{malschains.alpha}{Memetic Algorithm with Local
  Search Chains parameter. Crossover BLX-alpha. Lower
  values (<0.3) reduce diversity and a higher value
  increases diversity.}

  \item{malschains.threshold}{Memetic Algorithm with Local
  Search Chains parameter. Threshold that defines how much
  improvement in the local search is considered to be no
  improvement.}

  \item{malschains.maxEvals}{Memetic Algorithm with Local
  Search Chains parameter. Maximum number of evaluations.}

  \item{psoptim.maxit}{Particle Swarm Optimization
  parameter. Maximum number of iterations.}

  \item{psoptim.maxf}{Particle Swarm Optimization
  parameter. Maximum number of function evaluations.}

  \item{psoptim.abstol}{Particle Swarm Optimization
  parameter. Absolute convergence tolerance.}

  \item{psoptim.reltol}{Particle Swarm Optimization
  parameter. Tolerance for restarting.}

  \item{psoptim.s}{Particle Swarm Optimization parameter.
  Swarm size.}

  \item{psoptim.k}{Particle Swarm Optimization parameter.
  Exponent for calculating number of informants.}

  \item{psoptim.p}{Particle Swarm Optimization parameter.
  Average percentage of informants for each particle.}

  \item{psoptim.w}{Particle Swarm Optimization parameter.
  Exploitation constant.}

  \item{psoptim.c.p}{Particle Swarm Optimization parameter.
  Local exploration constant.}

  \item{psoptim.c.g}{Particle Swarm Optimization parameter.
  Global exploration constant.}

  \item{soma.pathLength}{Self-Organising Migrating
  Algorithm parameter. Distance (towards the leader) that
  individuals may migrate.}

  \item{soma.stepLength}{Self-Organising Migrating
  Algorithm parameter. Granularity at which potential steps
  are evaluated.}

  \item{soma.perturbationChance}{Self-Organising Migrating
  Algorithm parameter. Probability that individual
  parameters are changed on any given step.}

  \item{soma.minAbsoluteSep}{Self-Organising Migrating
  Algorithm parameter. Smallest absolute difference between
  maximum and minimum cost function values. Below this
  minimum the algorithm will terminate.}

  \item{soma.minRelativeSep}{Self-Organising Migrating
  Algorithm parameter. Smallest relative difference between
  maximum and minimum cost function values. Below this
  minimum the algorithm will terminate.}

  \item{soma.nMigrations}{Self-Organising Migrating
  Algorithm parameter. Maximum number of migrations to
  complete.}

  \item{soma.populationSize}{Self-Organising Migrating
  Algorithm parameter. Population size.}

  \item{tabu.iters}{Number of iterations in the preliminary
  search of the algorithm.}

  \item{tabu.listSize}{Tabu list size.}
}
\value{
  A list of class \code{hybridEnsemble} containing the
  following elements: \item{LR}{Bagged Logistic Regression
  model} \item{LR.lambda}{Shrinkage parameter}
  \item{RF}{Random Forest model} \item{AB}{Stochastic
  AdaBoost model} \item{KF}{Kernel Factory model}
  \item{NN}{Neural Network model} \item{SV}{Bagged Support
  Vector Machines model} \item{SB}{A label denoting which
  sub-ensemble was the single best}
  \item{weightsAUTHORITY}{The weights for the
  authority-based weighting method}
  \item{combine}{Combination methods used}
  \item{constants}{A vector denoting which predictors are
  constants} \item{minima}{Minimum values of the predictors
  required for preprocessing the data for the Neural
  Network} \item{scaling}{Range values of the predictors
  required for preprocessing the data for the Neural
  Network} \item{NumID}{Vector indicating which predictors
  are numeric} \item{calibratorLR}{The calibrator for the
  Bagged Logistic Regression model} \item{calibratorRF}{The
  calibrator for the Random Forest model}
  \item{calibratorAB}{The calibrator for the Stochastic
  AdaBoost model} \item{calibratorKF}{The calibrator for
  the Kernel Factory model} \item{calibratorNN}{The
  calibrator for the Neural Network model}
  \item{calibratorSV}{The calibrator for the Bagged Support
  Vector Machines model} \item{xVALIDATE}{Predictors of the
  validation sample } \item{predictions}{The seperate
  predictions by the six sub-ensembles}
  \item{yVALIDATE}{Response variable of the validation
  sample} \item{eval.measure}{The evaluation measure that
  was used}
}
\description{
  \code{hybridEnsemble} builds an ensemble consisting of
  six different sub-ensembles: Bagged Logistic Regressions,
  Random Forest, Stochastic AdaBoost, Kernel Factory,
  Bagged Neural Networks, Bagged Support Vector Machines.
}
\examples{
data(Credit)

\dontrun{
hE <-hybridEnsemble(x=Credit[1:100,names(Credit) != 'Response'],
                    y=Credit$Response[1:100],
                    RF.ntree=50,
                    AB.iter=50,
                    NN.size=5,
                    NN.decay=0,
                    SV.gamma = 2^-15,
                    SV.cost = 2^-5,
                    SV.degree=2,
                    SV.kernel='radial')
}
}
\author{
  Authors: Michel Ballings, Dauwe Vercamer, and Dirk Van
  den Poel, Maintainer: \email{Michel.Ballings@GMail.com}
}
\references{
  Ballings, M., Vercamer, D., Van den Poel, D., Hybrid
  Ensemble: Many Ensembles is Better Than One, Forthcoming.
}
\seealso{
  \code{\link{predict.hybridEnsemble}},
  \code{\link{importance.hybridEnsemble}},
  \code{\link{CVhybridEnsemble}},
  \code{\link{plot.CVhybridEnsemble}},
  \code{\link{summary.CVhybridEnsemble}}
}

