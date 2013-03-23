\name{plot.CVhybridEnsemble}
\alias{plot.CVhybridEnsemble}
\title{Plot the performance of the cross-validated Hybrid Ensemble}
\usage{
\method{plot}{CVhybridEnsemble}(x, y = NULL, ROCcurve = FALSE,
  averaging = "threshold", ...)
}
\arguments{
  \item{x}{An object of class CVhybridEnsemble}

  \item{y}{Not used}

  \item{ROCcurve}{TRUE or FALSE. Should the ROC curve be
  plotted or the median predictive performances?}

  \item{averaging}{For the ROC curve: "threshold"
  averaging, "horizontal" averaging, or "vertical"
  averaging.}

  \item{...}{Not used}
}
\description{
This function plots the averaged ROC curve per combination
method or the median predictive performance (Area under the
ROC, sensitivity or specificity curve depending on what was
used in the \code{CVhybridEnsemble} function).
}
\details{
In the legend: 'RBGA' (Genetic Algorithm),'DEOPT'
(Differential Evolution),'GENSA' (Generalized Simulated
Annealing),'MALSCHAINS' (Memetic Algorithm),'PSOPTIM'
(Particle Swarm),'SOMA' (Self Organizing Migrating
Algorithm),'TABU' (Tabue Search),'LHNNLS' (Lawson-Hanson
Non-negative least squares),'GINNLS' (Goldfarb-Idnani
Non-negative least squares),'NNloglik' (Non-negative
binomial likelihood),'MEAN' (Simple Mean),'SB' (Single
Best),'AUTHORITY' (Authority Based method). The letters
next to SB denote the single best for all cross-validation
runs: R=Random Forest, S=Bagged Support Vector Machines,
K=Kernel Factory, A=AdaBoost, L=Bagged Logistic Regression,
N=Bagged Neural Networks
}
\examples{
data(Credit)

\dontrun{
CVhE <- CVhybridEnsemble(x=Credit[1:200,names(Credit) != 'Response'],
                    y=Credit$Response[1:200],
                    verbose=TRUE,
                    RF.ntree=50,
                    KF.rp=1,
                    AB.iter=50,
                    NN.size=5,
                    NN.decay=0,
                    SV.gamma = 2^-15,
                    SV.cost = 2^-5,
                    SV.degree=2,
                    SV.kernel='radial')

plot(x=CVhE,ROCcurve= FALSE)
plot(x=CVhE,ROCcurve= TRUE)
}
}
\author{
Authors: Michel Ballings and Dirk Van den Poel, Maintainer:
\email{Michel.Ballings@GMail.com}
}
\references{
Ballings, M., Vercamer, D., Van den Poel, D., Hybrid
Ensemble: Many Ensembles is Better Than One, Forthcoming.
}
\seealso{
\code{\link{hybridEnsemble}},
\code{\link{predict.hybridEnsemble}},
\code{\link{importance.hybridEnsemble}},
\code{\link{CVhybridEnsemble}},
\code{\link{summary.CVhybridEnsemble}}
}

