\name{predict.hybridEnsemble}
\alias{predict.hybridEnsemble}
\title{Predict method for hybridEnsemble objects}
\usage{
  \method{predict}{hybridEnsemble} (object, newdata,
    verbose = FALSE, ...)
}
\arguments{
  \item{object}{An object of class hybridEnsemble created
  by the function \code{hybridEnsemble}}

  \item{newdata}{A data frame with the same predictors as
  in the training data}

  \item{verbose}{TRUE or FALSE. Should information be
  printed to the screen}

  \item{...}{Not currently used}
}
\value{
  A list containing the following vectors:
  \item{predMEAN}{Predictions combined by the simple mean}
  \item{SB}{A label denoting the single best algorithm:
  RF=Random Forest, LR= Bagged Logistic Regression, AB=
  AdaBoost, SV=Bagged Support Vector Machines, NN=Bagged
  Neural Networks, KF=Kernel Factory}
  \item{predSB}{Predictions by the single best}
  \item{predAUTHORITY}{Predictions combined by authority}
  ..and all the combination methods that are requested in
  the \code{\link{hybridEnsemble}} function.
}
\description{
  Prediction of new data using a hybridEnsemble model.
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

predictions <- predict(hE, newdata=Credit[1:100,names(Credit) != 'Response'])
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
  \code{\link{hybridEnsemble}},
  \code{\link{CVhybridEnsemble}},
  \code{\link{importance.hybridEnsemble}},
  \code{\link{plot.CVhybridEnsemble}},
  \code{\link{summary.CVhybridEnsemble}}
}

