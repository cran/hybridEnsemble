% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/summary.CVhybridEnsemble.R
\name{summary.CVhybridEnsemble}
\alias{summary.CVhybridEnsemble}
\title{Summarize the performance of the cross-validated Hybrid Ensemble}
\usage{
\method{summary}{CVhybridEnsemble}(
  object,
  name = "",
  stat = "median",
  LateX = FALSE,
  toppart = FALSE,
  bottompart = FALSE,
  all = TRUE,
  ...
)
}
\arguments{
\item{object}{An object of class CVhybridEnsemble}

\item{name}{Name of the dataset. Default is blank.}

\item{stat}{'median' or 'IQR' (inter quartile range) of the performance measure used in the CVhybridEnsemble object}

\item{LateX}{TRUE or FALSE. If true LateX code is printed to the screen. Otherwise a data frame.}

\item{toppart}{TRUE or FALSE. For the LateX table. Should the top part of the table be printed. Useful for concatenating multiple runs of the \code{summary} function (see examples).}

\item{bottompart}{TRUE or FALSE. For the LateX table. Should the bottom part of the table be printed. Useful for concatenating multiple runs of the \code{summary} function (see examples). The all parameter will not have effect when setting this to TRUE.}

\item{all}{TRUE or FALSE. Should the results of the predict.all be printed along with the main results? Works only when predict.all=TRUE in the call to \code{CVhybridEnsemble}.}

\item{...}{Not used}
}
\description{
This function produces summary results per combination method.
}
\details{
In the output: 'RBGA' (Genetic Algorithm), 'DEOPT' (Differential Evolution), 'GENSA' (Generalized Simulated Annealing), 'MALSCHAINS' (Memetic Algorithm), 'PSOPTIM' (Particle Swarm), 'SOMA' (Self Organizing Migrating Algorithm), 'TABU' (Tabue Search), 'LHNNLS' (Lawson-Hanson Non-negative least squares), 'GINNLS' (Goldfarb-Idnani Non-negative least squares), 'NNloglik' (Non-negative binomial likelihood), 'MEAN' (Simple Mean), 'SB' (Single Best), 'AUTHORITY' (Authority Based method). SB names denote the single best for all cross-validation runs: RF= Random Forest, SV= Bagged Support Vector Machines, KF= Kernel Factory, AB=AdaBoost, LR=Bagged Logistic Regression, NN=Bagged Neural Networks, RoF= Rotation Forest, KN= K-Nearest Neighbors.
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

summary(object=CVhE,stat='median')
summary(object=CVhE,stat='IQR')

#LaTeX table
#This code example shows how toppart and bottompart can be convenient if you want 
#to concatenate multiple datasets (here six time the same dataset).
#Paste the output of this code in your LateX document:
cat(
 summary(object=CVhE ,name="Credit", LateX=TRUE, toppart=TRUE),
 summary(object=CVhE ,name="Credit", LateX=TRUE),
 summary(object=CVhE, name="Credit", LateX=TRUE),
 summary(object=CVhE ,name="Credit", LateX=TRUE),
 summary(object=CVhE ,name="Credit", LateX=TRUE),
 summary(object=CVhE ,name="Credit", LateX=TRUE, bottompart=TRUE) )

}

}
\references{
Ballings, M., Vercamer, D., Bogaert, M., Van den Poel, D.
}
\seealso{
\code{\link{hybridEnsemble}}, \code{\link{predict.hybridEnsemble}}, \code{\link{importance.hybridEnsemble}}, \code{\link{CVhybridEnsemble}}, \code{\link{plot.CVhybridEnsemble}}
}
\author{
Michel Ballings, Dauwe Vercamer, Matthias Bogaert, and Dirk Van den Poel, Maintainer: \email{Michel.Ballings@GMail.com}
}
