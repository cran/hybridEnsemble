#' Five times twofold cross-validation for the Hybrid Ensemble function
#'
#' \code{CVhybridEnsemble} cross-validates (five times twofold) (\code{\link{hybridEnsemble}}) and computes performance statistics that can be plotted (\code{\link{plot.CVhybridEnsemble}}) and summarized (\code{\link{summary.CVhybridEnsemble}}).
#'
#' @param x A data frame of predictors. Categorical variables need to be transformed to binary (dummy) factors.
#' @param y A factor of observed class labels (responses) with the only allowed values \{0,1\}.,
#' @param algorithms Which algorihtms to use \{"LR","RF","AB","KF","NN","SV","RoF","KN","NB"\}. LR= Bagged Logistic Regression, RF=Random Forest, AB= AdaBoost, KF= Kernel Factory, NN= Bagged Neural Network, SV= Bagged Support Vector Machines, RoF= Rotation Forest, KN= Bagged K- Nearest Neighbors, NB= Bagged Naive Bayes.
#' @param combine Additional methods for combining the sub-ensembles. The simple mean, authority-based weighting and the single best are automatically provided since they are very effficient.  Possible additional methods: Genetic Algorithm: "rbga", Differential Evolutionary Algorithm: "DEopt", Generalized Simulated Annealing: "GenSA", Memetic Algorithm with Local Search Chains: "malschains", Particle Swarm Optimization: "psoptim", Self-Organising Migrating Algorithm: "soma", Tabu Search Algorithm: "tabu", Non-negative binomial likelihood: "NNloglik", Goldfarb-Idnani Non-negative least squares: "GINNLS", Lawson-Hanson Non-negative least squares: "LHNNLS".
#' @param eval.measure Evaluation measure for the following combination methods: authority-based method, single best, "rbga", "DEopt", "GenSA", "malschains", "psoptim", "soma", "tabu". Default is the area under the receiver operator characteristic curve 'auc'. The area under the sensitivity curve ('sens') and the area under the specificity curve ('spec') are also supported.
#' @param diversity TRUE or FALSE. Will set predict.all=TRUE in \code{hybridEnsemble} and compute diversity at the sub-ensemble and hybrid (i.e., meta) -ensemble level? Diversity is defined as 1 minus the absolute value of the mean of the pairwise correlations. The AUC will also be provided. For the AUC of the meta-ensemble the simple mean is used.
#' @param parallel TRUE or FALSE. Should the cross-validation be executed in parallel. Will use all available cores.
#' @param verbose TRUE or FALSE. Should information be printed to the screen while estimating the Hybrid Ensemble.
#' @param oversample TRUE or FALSE. Should oversampling be used? Setting oversample to TRUE helps avoid computational problems related to the subsetting process.
#' @param calibrate TRUE or FALSE. If FALSE percentile ranks of the prediction vectors will be used.
#' @param filter either NULL (deactivate) or a percentage denoting the minimum class size of dummy predictors. This parameter is used to remove near constants. For example if nrow(xTRAIN)=100, and filter=0.01 then all dummy predictors with any class size equal to 1 will be removed. Set this higher (e.g., 0.05 or 0.10) in case of errors.
#' @param LR.size Logistic Regression parameter. Ensemble size of the bagged logistic regression sub-ensemble.
#' @param RF.ntree Random Forest parameter. Number of trees to grow.
#' @param AB.iter Stochastic AdaBoost parameter. Number of boosting iterations to perform.
#' @param AB.maxdepth Stochastic AdaBoost parameter. The maximum depth of any node of the final tree, with the root node counted as depth 0.
#' @param KF.cp Kernel Factory parameter. The number of column partitions.
#' @param KF.rp Kernel Factory parameter. The number of row partitions.
#' @param KF.ntree Kernel Factory parameter. Number of trees to grow.
#' @param NN.rang Neural Network parameter. Initial random weights on [-rang, rang].
#' @param NN.maxit Neural Network parameter. Maximum number of iterations.
#' @param NN.size Neural Network parameter. Number of units in the single hidden layer. Can be mutiple values that need to be optimized.
#' @param NN.decay Neural Network parameter. Weight decay. Can be mutiple values that need to be optimized.
#' @param NN.skip Neural Network parameter. Switch to add skip-layer connections from input to output. Can be boolean vector (TRUE and FALSE) for optimization.
#' @param NN.ens.size Neural Network parameter. Ensemble size of the neural network sub-ensemble.
#' @param SV.gamma Support Vector Machines parameter. Width of the Guassian for radial basis and sigmoid kernel. Can be mutiple values that need to be optimized.
#' @param SV.cost Support Vector Machines parameter. Penalty (soft margin constant). Can be mutiple values that need to be optimized.
#' @param SV.degree Support Vector Machines parameter. Degree of the polynomial kernel. Can be mutiple values that need to be optimized.
#' @param SV.kernel Support Vector Machines parameter. Kernels to try. Can be one or more of: 'radial','sigmoid','linear','polynomial'. Can be mutiple values that need to be optimized.
#' @param SV.size Support Vector Machines parameter. Ensemble size of the SVM sub-ensemble.
#' @param RoF.L Rotation Forest parameter. Number of trees to grow.
#' @param KN.K K-Nearest Neighbors parameter. Number of nearest neighbors to try. For example c(10,20,30). The optimal K will be selected. If larger than nrow(xTRAIN) the maximum K will be reset to 50\% of nrow(xTRAIN). Can be mutiple values that need to be optimized.
#' @param KN.size K-Nearest Neighbors parameter. Ensemble size of the K-nearest neighbor sub-ensemble.
#' @param NB.size Naive Bayes parameter. Ensemble size of the bagged naive bayes sub-ensemble.
#' @param rbga.popSize Genetic Algorithm parameter. Population size. Default is 14 times the number of variables.
#' @param rbga.iters Genetic Algorithm parameter.  Number of iterations.
#' @param rbga.mutationChance Genetic Algorithm parameter. The chance that a gene in the chromosome mutates.
#' @param rbga.elitism Genetic Algorithm parameter. Number of chromosomes that are kept into the next generation.
#' @param DEopt.nP Differential Evolutionary Algorithm parameter. Population size.
#' @param DEopt.nG Differential Evolutionary Algorithm parameter. Number of generations.
#' @param DEopt.F Differential Evolutionary Algorithm parameter. Step size.
#' @param DEopt.CR Differential Evolutionary Algorithm parameter. Probability of crossover.
#' @param GenSA.maxit Generalized Simulated Annealing. Maximum number of iterations.
#' @param GenSA.temperature Generalized Simulated Annealing. Initial value for temperature.
#' @param GenSA.visiting.param Generalized Simulated Annealing. Parameter for visiting distribution.
#' @param GenSA.acceptance.param Generalized Simulated Annealing. Parameter for acceptance distribution.
#' @param GenSA.max.call Generalized Simulated Annealing. Maximum number of calls of the objective function.
#' @param malschains.popsize Memetic Algorithm with Local Search Chains parameter. Population size.
#' @param malschains.ls Memetic Algorithm with Local Search Chains parameter. Local search method.
#' @param malschains.istep Memetic Algorithm with Local Search Chains parameter. Number of iterations of the local search.
#' @param malschains.effort Memetic Algorithm with Local Search Chains parameter. Value between 0 and 1. The ratio between the number of evaluations for the local search and for the evolutionary algorithm. A higher effort means more evaluations for the evolutionary algorithm.
#' @param malschains.alpha Memetic Algorithm with Local Search Chains parameter. Crossover BLX-alpha. Lower values (<0.3) reduce diversity and a higher value increases diversity.
#' @param malschains.threshold Memetic Algorithm with Local Search Chains parameter. Threshold that defines how much improvement in the local search is considered to be no improvement.
#' @param malschains.maxEvals Memetic Algorithm with Local Search Chains parameter. Maximum number of evaluations.
#' @param psoptim.maxit Particle Swarm Optimization parameter. Maximum number of iterations.
#' @param psoptim.maxf Particle Swarm Optimization parameter. Maximum number of function evaluations.
#' @param psoptim.abstol Particle Swarm Optimization parameter. Absolute convergence tolerance.
#' @param psoptim.reltol Particle Swarm Optimization parameter. Tolerance for restarting.
#' @param psoptim.s Particle Swarm Optimization parameter. Swarm size.
#' @param psoptim.k Particle Swarm Optimization parameter. Exponent for calculating number of informants.
#' @param psoptim.p Particle Swarm Optimization parameter. Average percentage of informants for each particle.
#' @param psoptim.w Particle Swarm Optimization parameter. Exploitation constant.
#' @param psoptim.c.p Particle Swarm Optimization parameter. Local exploration constant.
#' @param psoptim.c.g Particle Swarm Optimization parameter. Global exploration constant.
#' @param soma.pathLength Self-Organising Migrating Algorithm parameter. Distance (towards the leader) that individuals may migrate.
#' @param soma.stepLength Self-Organising Migrating Algorithm parameter. Granularity at which potential steps are evaluated.
#' @param soma.perturbationChance Self-Organising Migrating Algorithm parameter. Probability that individual parameters are changed on any given step.
#' @param soma.minAbsoluteSep Self-Organising Migrating Algorithm parameter. Smallest absolute difference between maximum and minimum cost function values. Below this minimum the algorithm will terminate.
#' @param soma.minRelativeSep Self-Organising Migrating Algorithm parameter. Smallest relative difference between maximum and minimum cost function values. Below this minimum the algorithm will terminate.
#' @param soma.nMigrations Self-Organising Migrating Algorithm parameter. Maximum number of migrations to complete.
#' @param soma.populationSize Self-Organising Migrating Algorithm parameter. Population size.
#' @param tabu.iters Number of iterations in the preliminary search of the algorithm.
#' @param tabu.listSize Tabu list size.
#'
#'
#' @examples
#'
#' data(Credit)
#'
#' \dontrun{
#' x <- Credit[1:200,names(Credit) != 'Response']
#' x <- x[,sapply(x,is.numeric)]
#' CVhE <- CVhybridEnsemble(x=x,
#'                     y=Credit$Response[1:200],
#'                     verbose=TRUE,
#'                     KF.rp=1,
#'                     RF.ntree=50,
#'                     AB.iter=50,
#'                     NN.size=5,
#'                     NN.decay=0,
#'                     SV.gamma = 2^-15,
#'                     SV.cost = 2^-5,
#'                     SV.degree=2,
#'                     SV.kernel='radial')
#' }
#' @references Ballings, M., Vercamer, D., Bogaert, M., Van den Poel, D.
#' @seealso \code{\link{hybridEnsemble}}, \code{\link{predict.hybridEnsemble}}, \code{\link{importance.hybridEnsemble}}, \code{\link{plot.CVhybridEnsemble}}, \code{\link{summary.CVhybridEnsemble}}
#' @return A list of class \code{CVhybridEnsemble} containing the following elements:
#' \item{MEAN}{For the simple mean combination method: A list containing the median and inter quartile range of the performance evaluations, the performance evaluations on each fold, and the predictions and reponse vectors for each fold.}
#' \item{AUTHORITY}{For the authority combination method: A list containing the median and inter quartile range of the performance evaluations, the performance evaluations on each fold, and the predictions and reponse vectors for each fold.}
#' \item{SB}{For the single best: A list containing the median and inter quartile range of the performance evaluations, the performance evaluations on each fold, and the predictions and reponse vectors for each fold.}
#' ..and all the combination methods that are requested.
#' \item{eval.measure}{The performance measure that was used} 
#' \item{diversity}{Data frame containing the diversity (1 minus the absolute value of the mean of the pairwise correlations), and mean auc and accuracy(threshold=0.5) of the hybrid ensemble and the sub-ensembles.}  
#' @author Michel Ballings, Dauwe Vercamer, Matthias Bogaert, and Dirk Van den Poel, Maintainer: \email{Michel.Ballings@@GMail.com}

CVhybridEnsemble <- function(x=NULL,
                             y=NULL,
                             algorithms=c("LR","RF","AB","KF","NN","SV","RoF","KN","NB"),
                             combine=NULL,
                             eval.measure='auc',
                             diversity=FALSE,
                             parallel=FALSE,
                             verbose=FALSE,
                             oversample=TRUE,
                             calibrate=FALSE,
                             filter= 0.03,
                             LR.size=10,
                             RF.ntree=500,
                             AB.iter=500,
                             AB.maxdepth=3,
                             KF.cp=1,
                             KF.rp=round(log(nrow(x),10)),
                             KF.ntree=500,
                             NN.rang=0.1,
                             NN.maxit=10000,
                             NN.size=c(5,10,20),
                             NN.decay=c(0,0.001,0.01,0.1),
                             NN.skip=c(TRUE,FALSE),  
                             NN.ens.size=10,
                             SV.gamma = 2^(-15:3),
                             SV.cost = 2^(-5:13),
                             SV.degree=c(2,3),
                             SV.kernel=c('radial','sigmoid','linear','polynomial'),
                             SV.size=10,
                             RoF.L=10,
                             KN.K=c(1:150),
                             KN.size=10,
                             NB.size=10,
                             rbga.popSize = length(algorithms)*14,
                             rbga.iters = 500,
                             rbga.mutationChance = 1/ rbga.popSize,
                             rbga.elitism= max(1, round(rbga.popSize*0.05)) ,
                             DEopt.nP=20,
                             DEopt.nG=500,
                             DEopt.F=0.9314,
                             DEopt.CR=0.6938,
                             GenSA.maxit=500,
                             GenSA.temperature=0.5,
                             GenSA.visiting.param=2.7,
                             GenSA.acceptance.param=-5,
                             GenSA.max.call=1e7,
                             malschains.popsize=60,
                             malschains.ls="cmaes",
                             malschains.istep=300,
                             malschains.effort=0.5,
                             malschains.alpha=0.5,
                             malschains.threshold=1e-08,
                             malschains.maxEvals=500,
                             psoptim.maxit=500,
                             psoptim.maxf=Inf,
                             psoptim.abstol=-Inf,
                             psoptim.reltol=0,
                             psoptim.s=40,
                             psoptim.k=3,
                             psoptim.p=1-(1-1/psoptim.s)^psoptim.k,
                             psoptim.w=1/(2*log(2)),
                             psoptim.c.p=.5+log(2),
                             psoptim.c.g=.5+log(2),
                             soma.pathLength=3,
                             soma.stepLength=0.11,
                             soma.perturbationChance=0.1,
                             soma.minAbsoluteSep=0,
                             soma.minRelativeSep= 0.001,
                             soma.nMigrations=500,
                             soma.populationSize=10,
                             tabu.iters=500,
                             tabu.listSize=c(5:12)
                             ){


  object <- call('hybridEnsemble',
                 algorithms=algorithms,
                 combine=combine,
                 eval.measure=eval.measure,
                 verbose=FALSE,
                 oversample=oversample,
                 filter= filter,
                 LR.size=LR.size,
                 RF.ntree=RF.ntree,
                 AB.iter=AB.iter,
                 AB.maxdepth=AB.maxdepth,
                 KF.cp=KF.cp,
                 KF.rp=KF.rp,
                 KF.ntree=KF.ntree,
                 NN.rang=NN.rang,
                 NN.maxit=NN.maxit,
                 NN.size=NN.size,
                 NN.decay=NN.decay,
                 NN.skip=NN.skip,  
                 NN.ens.size=NN.ens.size,
                 SV.gamma = SV.gamma,
                 SV.cost = SV.cost,
                 SV.degree=SV.degree,
                 SV.kernel=SV.kernel,
                 SV.size=SV.size,
                 RoF.L=RoF.L,
                 KN.K=KN.K,
                 KN.size=KN.size,
                 NB.size=NB.size,
                 rbga.popSize = rbga.popSize,
                 rbga.iters = rbga.iters,
                 rbga.mutationChance = rbga.mutationChance,
                 rbga.elitism= rbga.elitism ,
                 DEopt.nP=DEopt.nP,
                 DEopt.nG=DEopt.nG,
                 DEopt.F=DEopt.F,
                 DEopt.CR=DEopt.CR,
                 GenSA.maxit=GenSA.maxit,
                 GenSA.temperature=GenSA.temperature,
                 GenSA.visiting.param=GenSA.visiting.param,
                 GenSA.acceptance.param=GenSA.acceptance.param,
                 GenSA.max.call=GenSA.max.call,
                 malschains.popsize=malschains.popsize,
                 malschains.ls=malschains.ls,
                 malschains.istep=malschains.istep,
                 malschains.effort=malschains.effort,
                 malschains.alpha=malschains.alpha,
                 malschains.threshold=malschains.threshold,
                 malschains.maxEvals=malschains.maxEvals,
                 psoptim.maxit=psoptim.maxit,
                 psoptim.maxf=psoptim.maxf,
                 psoptim.abstol=psoptim.abstol,
                 psoptim.reltol=psoptim.reltol,
                 psoptim.s=psoptim.s,
                 psoptim.k=psoptim.k,
                 psoptim.p=psoptim.p,
                 psoptim.w=psoptim.w,
                 psoptim.c.p=psoptim.c.p,
                 psoptim.c.g=psoptim.c.g,
                 soma.pathLength=soma.pathLength,
                 soma.stepLength=soma.stepLength,
                 soma.perturbationChance=soma.perturbationChance,
                 soma.minAbsoluteSep=soma.minAbsoluteSep,
                 soma.minRelativeSep= soma.minRelativeSep,
                 soma.nMigrations=soma.nMigrations,
                 soma.populationSize=soma.populationSize,
                 tabu.iters=tabu.iters,
                 tabu.listSize=tabu.listSize
                 )
  tab <- table(y)
  if (!all(tab==tab[1])){
      if (oversample) {

        #oversample instances from the smallest class
        whichmin <- which(y==as.integer(names(which.min(tab))))
        indmin <- sample(whichmin,max(tab),replace=TRUE)
        indmin <- c(whichmin,indmin)[1:max(tab)]
        #take all the instances of the dominant class
        indmax <- which(y==as.integer(names(which.max(tab))))
        x <- x[c(indmin,indmax),]
        y <- y[c(indmin,indmax)]
      }
  }

  folds <- .partition(y,p=0.5,times=5)
  
  folds2 <- train <- test <- list()
  for (i in 1:length(folds)){
    train[[i]] <- folds[[i]]$train
    test[[i]] <- folds[[i]]$test
  }
  folds2 <- c(train,test)

  perf <- numeric(length(folds2))
  
  result <- list()
  if (tolower('rbga') %in% tolower(object$combine)) result$RBGA <- perf
  if (tolower('DEopt') %in% tolower(object$combine)) result$DEOPT <- perf
  if (tolower('GenSA') %in% tolower(object$combine)) result$GENSA <- perf
  if (tolower('malschains') %in% tolower(object$combine)) result$MALSCHAINS <- perf
  if (tolower('psoptim') %in% tolower(object$combine)) result$PSOPTIM <- perf
  if (tolower('soma') %in% tolower(object$combine)) result$SOMA <- perf
  if (tolower('tabu') %in% tolower(object$combine)) result$TABU <- perf

  if (tolower('LHNNLS') %in% tolower(object$combine)) result$LHNNLS <- perf
  if (tolower('GINNLS') %in% tolower(object$combine)) result$GINNLS <- perf
  if (tolower('NNloglik') %in% tolower(object$combine)) result$NNloglik <- perf


  result$MEAN <- perf
  result$AUTHORITY <- perf
  result$SB <- perf

  SBname <- perf

  predANDresp <- list()
  if (tolower('rbga') %in% tolower(object$combine)) predANDresp$RBGA <- list()
  if (tolower('DEopt') %in% tolower(object$combine)) predANDresp$DEOPT <- list()
  if (tolower('GenSA') %in% tolower(object$combine)) predANDresp$GENSA <- list()
  if (tolower('malschains') %in% tolower(object$combine)) predANDresp$MALSCHAINS <- list()
  if (tolower('psoptim') %in% tolower(object$combine)) predANDresp$PSOPTIM <- list()
  if (tolower('soma') %in% tolower(object$combine)) predANDresp$SOMA <- list()
  if (tolower('tabu') %in% tolower(object$combine)) predANDresp$TABU <- list()


  if (tolower('LHNNLS') %in% tolower(object$combine)) predANDresp$LHNNLS <- list()
  if (tolower('GINNLS') %in% tolower(object$combine)) predANDresp$GINNLS <- list()
  if (tolower('NNloglik') %in% tolower(object$combine)) predANDresp$NNloglik <- list()


  predANDresp$MEAN <- list()
  predANDresp$AUTHORITY <- list()
  predANDresp$SB <- list()

  divstore <- data.frame()
  
if (!parallel){


   for (i in 1:length(folds2)){




      if (verbose==TRUE) {
        if (i==1) {
          start <- Sys.time()
          cat(paste("Fold ", i))
        }
        if (i==2)  cat(paste(' /Estimated time of completion: ', Sys.time() + diff*9),"\n" )
        if (i >= 2)  cat(paste("Fold ", i),"\n")

      }

      xTRAIN <- x[ folds2[[i]], ]
      yTRAIN <- y[ folds2[[i]]  ]
      xTEST <-  x[ -folds2[[i]], ]
      yTEST <-  y[ -folds2[[i]]  ]

      ## Already taken care of by the benchmarkR function, do not uncomment!
      #remove constants and near constants
      # constants <- sapply(xTRAIN,function(x){all(as.numeric(x[1])==as.numeric(x))})
      # if (!is.null(filter)) constants <- sapply(xTRAIN,function(x) (length(unique(x))==2 && any(table(x) <= round(nrow(xTRAIN)*filter))) || all(as.numeric(x[1])==as.numeric(x))   )
      # xTRAIN <- xTRAIN[,!constants]
      # xTEST <- xTEST[,!constants]


      Call <- c(as.list(object), x=list(xTRAIN), y=list(yTRAIN))
      model <- eval(as.call(Call))

      SBname[i] <- model$SB

      pred <- predict(model, xTEST, predict.all=diversity)

      #Compute diversity
      if (diversity==TRUE){

        div <- data.frame(matrix(NA,nrow=length(names(pred$baseclassifiers))+1,ncol=4))
        colnames(div) <- c("ensemble","diversity","auc","accuracy")
        jj <- 0
        for (j in names(pred$baseclassifiers)) {
          jj <- jj+1
          div[jj,] <- c(j,.diversity(pred$baseclassifiers[[j]]),
                           AUC::auc(AUC::roc(pred$subensemble[[substring(j,5)]],yTEST)),
                           AUC::accuracy(pred$subensemble[[substring(j,5)]],yTEST)$measure[which.min(abs(accuracy(pred$subensemble[[substring(j,5)]],yTEST)$cutoffs-0.5))])
        }
        diver <- .diversity(pred$subensemble) #correlations are not impacted by weighting
        for (iii in grep("pred",names(pred),value=TRUE)[!grep("pred",names(pred),value=TRUE) %in% "predSB"]){
            jj <- jj + 1
            div[jj,] <- c(iii, diver,
                          AUC::auc(AUC::roc(pred[[iii]],yTEST)),
                          AUC::accuracy(pred[[iii]],yTEST)$measure[which.min(abs(accuracy(pred[[iii]],yTEST)$cutoffs-0.5))])
          
        }
        
        div[nrow(div)+1,] <- c("predSB",div[div$ensemble==paste0("pred",pred$SB),2:4])
        
        div[,2:4] <- sapply(div[,2:4],as.numeric)
        divstore <- rbind(divstore,div)

      }
      #end diversity computation

      if (tolower(object$eval.measure)=='spec') {



                      if (!is.null(pred$predRBGA)) {

                        result$RBGA[i] <-  AUC::auc(specificity(pred$predRBGA,yTEST))
                        predANDresp$RBGA[[i]] <- list(predicted=as.numeric(pred$predRBGA),response=yTEST)
                      }

                      if (!is.null(pred$predDEOPT)){

                        result$DEOPT[i] <- AUC::auc(specificity(pred$predDEOPT ,yTEST))
                        predANDresp$DEOPT[[i]] <- list(predicted=as.numeric(pred$predDEOPT),response=yTEST)
                      }

                      if (!is.null(pred$predGENSA)) {

                        result$GENSA[i] <- AUC::auc(specificity( pred$predGENSA,yTEST))
                        predANDresp$GENSA[[i]] <- list(predicted=as.numeric(pred$predGENSA),response=yTEST)
                      }

                      if (!is.null(pred$predMALSCHAINS)) {

                        result$MALSCHAINS[i] <-  AUC::auc(specificity(pred$predMALSCHAINS ,yTEST))
                        predANDresp$MALSCHAINS[[i]] <- list(predicted=as.numeric(pred$predMALSCHAINS),response=yTEST)
                      }

                      if (!is.null(pred$predPSOPTIM)){

                        result$PSOPTIM[i] <-  AUC::auc(specificity( pred$predPSOPTIM,yTEST))
                        predANDresp$PSOPTIM[[i]] <- list(predicted=as.numeric(pred$predPSOPTIM),response=yTEST)
                      }

                      if (!is.null(pred$predSOMA)) {

                        result$SOMA[i] <-  AUC::auc(specificity( pred$predSOMA,yTEST))
                        predANDresp$SOMA[[i]] <- list(predicted=as.numeric(pred$predSOMA),response=yTEST)
                      }

                      if (!is.null(pred$predTABU)) {

                        result$TABU[i] <-  AUC::auc(specificity(pred$predTABU ,yTEST))
                        predANDresp$TABU[[i]] <- list(predicted=as.numeric(pred$predTABU),response=yTEST)
                      }


                      if (!is.null(pred$predLHNNLS)) {

                        result$LHNNLS[i] <-  AUC::auc(specificity(pred$predLHNNLS ,yTEST))
                        predANDresp$LHNNLS[[i]] <- list(predicted=as.numeric(pred$predLHNNLS),response=yTEST)
                      }
                      if (!is.null(pred$predGINNLS)) {

                        result$GINNLS[i] <- AUC::auc(specificity(pred$predGINNLS ,yTEST))
                        predANDresp$GINNLS[[i]] <- list(predicted=as.numeric(pred$predGINNLS),response=yTEST)

                      }
                      if (!is.null(pred$predNNloglik)) {

                        result$NNloglik[i] <-  AUC::auc(specificity(pred$predNNloglik ,yTEST))
                        predANDresp$NNloglik[[i]] <- list(predicted=as.numeric(pred$predNNloglik),response=yTEST)
                      }





                      result$MEAN[i] <-  AUC::auc(specificity(pred$predMEAN ,yTEST))
                      predANDresp$MEAN[[i]] <- list(predicted=as.numeric(pred$predMEAN),response=yTEST)



                      result$AUTHORITY[i] <-  AUC::auc(specificity(pred$predAUTHORITY ,yTEST))
                      predANDresp$AUTHORITY[[i]] <- list(predicted=as.numeric(pred$predAUTHORITY),response=yTEST)



                      result$SB[i] <-  AUC::auc(specificity(pred$predSB ,yTEST))
                      predANDresp$SB[[i]] <- list(predicted=as.numeric(pred$predSB),response=yTEST)


      } else if (tolower(object$eval.measure)=='sens') {

                      if (!is.null(pred$predRBGA)) {

                        result$RBGA[i] <- AUC::auc(sensitivity(pred$predRBGA ,yTEST))
                        predANDresp$RBGA[[i]] <- list(predicted=as.numeric(pred$predRBGA),response=yTEST)
                      }

                      if (!is.null(pred$predDEOPT)){

                        result$DEOPT[i] <- AUC::auc(sensitivity(pred$predDEOPT ,yTEST))
                        predANDresp$DEOPT[[i]] <- list(predicted=as.numeric(pred$predDEOPT),response=yTEST)
                      }

                      if (!is.null(pred$predGENSA)) {

                        result$GENSA[i] <- AUC::auc(sensitivity(pred$predGENSA ,yTEST))
                        predANDresp$GENSA[[i]] <- list(predicted=as.numeric(pred$predGENSA),response=yTEST)
                      }

                      if (!is.null(pred$predMALSCHAINS)) {

                        result$MALSCHAINS[i] <- AUC::auc(sensitivity(pred$predMALSCHAINS ,yTEST))
                        predANDresp$MALSCHAINS[[i]] <- list(predicted=as.numeric(pred$predMALSCHAINS),response=yTEST)
                      }

                      if (!is.null(pred$predPSOPTIM)){

                        result$PSOPTIM[i] <- AUC::auc(sensitivity(pred$predPSOPTIM ,yTEST))
                        predANDresp$PSOPTIM[[i]] <- list(predicted=as.numeric(pred$predPSOPTIM),response=yTEST)
                      }

                      if (!is.null(pred$predSOMA)) {

                        result$SOMA[i] <-  AUC::auc(sensitivity(pred$predSOMA ,yTEST))
                        predANDresp$SOMA[[i]] <- list(predicted=as.numeric(pred$predSOMA),response=yTEST)
                      }

                      if (!is.null(pred$predTABU)) {

                        result$TABU[i] <-  AUC::auc(sensitivity(pred$predTABU ,yTEST))
                        predANDresp$TABU[[i]] <- list(predicted=as.numeric(pred$predTABU),response=yTEST)
                      }


                      if (!is.null(pred$predLHNNLS)) {

                        result$LHNNLS[i] <- AUC::auc(sensitivity(pred$predLHNNLS ,yTEST))
                        predANDresp$LHNNLS[[i]] <- list(predicted=as.numeric(pred$predLHNNLS),response=yTEST)
                      }
                      if (!is.null(pred$predGINNLS)) {

                        result$GINNLS[i] <-  AUC::auc(sensitivity(pred$predGINNLS ,yTEST))
                        predANDresp$GINNLS[[i]] <- list(predicted=as.numeric(pred$predGINNLS),response=yTEST)

                      }
                      if (!is.null(pred$predNNloglik)) {

                        result$NNloglik[i] <-  AUC::auc(sensitivity(pred$predNNloglik ,yTEST))
                        predANDresp$NNloglik[[i]] <- list(predicted=as.numeric(pred$predNNloglik),response=yTEST)
                      }





                      result$MEAN[i] <-  AUC::auc(sensitivity(pred$predMEAN ,yTEST))
                      predANDresp$MEAN[[i]] <- list(predicted=as.numeric(pred$predMEAN),response=yTEST)



                      result$AUTHORITY[i] <-  AUC::auc(sensitivity(pred$predAUTHORITY ,yTEST))
                      predANDresp$AUTHORITY[[i]] <- list(predicted=as.numeric(pred$predAUTHORITY),response=yTEST)



                      result$SB[i] <-  AUC::auc(sensitivity(pred$predSB ,yTEST))
                      predANDresp$SB[[i]] <- list(predicted=as.numeric(pred$predSB),response=yTEST)





      } else  if (tolower(object$eval.measure)=='auc') {



                      if (!is.null(pred$predRBGA)) {
                        result$RBGA[i] <- AUC::auc(roc( pred$predRBGA,yTEST))
                        predANDresp$RBGA[[i]] <- list(predicted=as.numeric(pred$predRBGA),response=yTEST)
                      }

                      if (!is.null(pred$predDEOPT)){
                        result$DEOPT[i] <- AUC::auc(roc(pred$predDEOPT,yTEST))
                        predANDresp$DEOPT[[i]] <- list(predicted=as.numeric(pred$predDEOPT),response=yTEST)
                      }

                      if (!is.null(pred$predGENSA)) {
                        result$GENSA[i] <- AUC::auc(roc(pred$predGENSA ,yTEST))
                        predANDresp$GENSA[[i]] <- list(predicted=as.numeric(pred$predGENSA),response=yTEST)
                      }

                      if (!is.null(pred$predMALSCHAINS)) {
                        result$MALSCHAINS[i] <- AUC::auc(roc(pred$predMALSCHAINS ,yTEST))
                        predANDresp$MALSCHAINS[[i]] <- list(predicted=as.numeric(pred$predMALSCHAINS),response=yTEST)
                      }

                      if (!is.null(pred$predPSOPTIM)){
                        result$PSOPTIM[i] <- AUC::auc(roc(pred$predPSOPTIM ,yTEST))
                        predANDresp$PSOPTIM[[i]] <- list(predicted=as.numeric(pred$predPSOPTIM),response=yTEST)
                      }

                      if (!is.null(pred$predSOMA)) {
                        result$SOMA[i] <- AUC::auc(roc(pred$predSOMA ,yTEST))
                        predANDresp$SOMA[[i]] <- list(predicted=as.numeric(pred$predSOMA),response=yTEST)
                      }

                      if (!is.null(pred$predTABU)) {
                        result$TABU[i] <- AUC::auc(roc(pred$predTABU ,yTEST))
                        predANDresp$TABU[[i]] <- list(predicted=as.numeric(pred$predTABU),response=yTEST)
                      }



                      if (!is.null(pred$predLHNNLS)) {
                        result$LHNNLS[i] <- AUC::auc(roc(pred$predLHNNLS ,yTEST))
                        predANDresp$LHNNLS[[i]] <- list(predicted=as.numeric(pred$predLHNNLS),response=yTEST)
                      }
                      if (!is.null(pred$predGINNLS)) {
                        result$GINNLS[i] <- AUC::auc(roc(pred$predGINNLS ,yTEST))
                        predANDresp$GINNLS[[i]] <- list(predicted=as.numeric(pred$predGINNLS),response=yTEST)

                      }
                      if (!is.null(pred$predNNloglik)) {
                        result$NNloglik[i] <- AUC::auc(roc(pred$predNNloglik ,yTEST))
                        predANDresp$NNloglik[[i]] <- list(predicted=as.numeric(pred$predNNloglik),response=yTEST)
                      }



                      result$MEAN[i] <- AUC::auc(roc(pred$predMEAN ,yTEST))
                      predANDresp$MEAN[[i]] <- list(predicted=as.numeric(pred$predMEAN),response=yTEST)

                      result$AUTHORITY[i] <- AUC::auc(roc( pred$predAUTHORITY,yTEST))
                      predANDresp$AUTHORITY[[i]] <- list(predicted=as.numeric(pred$predAUTHORITY),response=yTEST)

                      result$SB[i] <- AUC::auc(roc( pred$predSB,yTEST))
                      predANDresp$SB[[i]] <- list(predicted=as.numeric(pred$predSB),response=yTEST)

      }


      

    if (verbose==TRUE) {
        if (i==1) diff <- Sys.time() - start
      }

      }


} else if (parallel){ 
  
   registerDoParallel(detectCores())

      if (verbose==TRUE) {
      cat("Number of workers:", getDoParWorkers()," \n")
      cat("Parallel backend name:", getDoParName()," \n")
      cat("Parallel backend version:", getDoParVersion()," \n")
      }

      predAndDiv <- foreach(i = seq(folds2), .inorder=TRUE, .verbose=verbose, .errorhandling="stop",
                      .packages = c("randomForest", "kernelFactory", "ada", "rpart", "nnet", "e1071", "NMOF", "GenSA", "Rmalschains", "pso", "AUC", "soma", "genalg", "reportr", "nnls", "quadprog", "tabuSearch", "rotationForest", "FNN", "glmnet")) %dopar%
            {
  
                        xTRAIN <- x[ folds2[[i]], ]
                        yTRAIN <- y[ folds2[[i]]  ]
                        xTEST <-  x[ -folds2[[i]], ]
                        yTEST <-  y[ -folds2[[i]]  ]
                  
                        
                        ## Already taken care of by the benchmarkR function, do not uncomment!
                        #remove constants and near constants
                        # constants <- sapply(xTRAIN,function(x){all(as.numeric(x[1])==as.numeric(x))})
                        # if (!is.null(filter)) constants <- sapply(xTRAIN,function(x) (length(unique(x))==2 && any(table(x) <= round(nrow(xTRAIN)*filter))) || all(as.numeric(x[1])==as.numeric(x))   )
                        # xTRAIN <- xTRAIN[,!constants]
                        # xTEST <- xTEST[,!constants]
                        
                  
                        Call <- c(as.list(object), x=list(xTRAIN), y=list(yTRAIN))
                        model <- eval(as.call(Call))
                  
                        SBname <- model$SB
                  
                        pred <- predict(model, xTEST, predict.all=diversity)
                        
                        div <- NULL
                        
                        
                        
                        # #Compute diversity
                        # if (diversity==TRUE){
                        # 
                        #   div <- data.frame(matrix(NA,nrow=length(names(pred$baseclassifiers))+1,ncol=4))
                        #   colnames(div) <- c("ensemble","diversity","auc","accuracy")
                        #   jj <- 0
                        #   for (j in names(pred$baseclassifiers)) {
                        #     jj <- jj+1
                        #     div[jj,] <- c(j,.diversity(pred$baseclassifiers[[j]]),
                        #                      AUC::auc(AUC::roc(pred$subensemble[[substring(j,5)]],yTEST)),
                        #                      AUC::accuracy(pred$subensemble[[substring(j,5)]],yTEST)$measure[which.min(abs(accuracy(pred$subensemble[[substring(j,5)]],yTEST)$cutoffs-0.5))])
                        #   }
                        #   diver <- .diversity(pred$subensemble)
                        #   for (iii in grep("pred",names(pred),value=TRUE)){
                        #       jj <- jj + 1
                        #       div[jj,] <- c(iii, diver,
                        #                     AUC::auc(AUC::roc(pred[[iii]],yTEST)),
                        #                     AUC::accuracy(pred[[iii]],yTEST)$measure[which.min(abs(accuracy(pred[[iii]],yTEST)$cutoffs-0.5))])
                        #     
                        #   }
                        #   div[,2:4] <- sapply(div[,2:4],as.numeric)
                        #   div
                        # 
                        # }
                        # #end diversity computation
                        
                        
                        #Compute diversity
                        
                        if (diversity==TRUE){
                  
                          div <- data.frame(matrix(NA,nrow=length(names(pred$baseclassifiers))+1,ncol=4))
                          colnames(div) <- c("ensemble","diversity","auc","accuracy")
                          jj <- 0
                          for (j in names(pred$baseclassifiers)) {
                            jj <- jj+1
                            div[jj,] <- c(j,.diversity(pred$baseclassifiers[[j]]),
                                             AUC::auc(AUC::roc(pred$subensemble[[substring(j,5)]],yTEST)),
                                             AUC::accuracy(pred$subensemble[[substring(j,5)]],yTEST)$measure[which.min(abs(accuracy(pred$subensemble[[substring(j,5)]],yTEST)$cutoffs-0.5))])
                          }
                          diver <- .diversity(pred$subensemble) #correlations are not impacted by weighting
                          for (iii in grep("pred",names(pred),value=TRUE)[!grep("pred",names(pred),value=TRUE) %in% "predSB"]){
                              jj <- jj + 1
                              div[jj,] <- c(iii, diver,
                                            AUC::auc(AUC::roc(pred[[iii]],yTEST)),
                                            AUC::accuracy(pred[[iii]],yTEST)$measure[which.min(abs(accuracy(pred[[iii]],yTEST)$cutoffs-0.5))])
                            
                          }
                          
                          div[nrow(div)+1,] <- c("predSB",div[div$ensemble==paste0("pred",pred$SB),2:4])
                          
                          div[,2:4] <- sapply(div[,2:4],as.numeric)
                          divstore <- rbind(divstore,div)
                  
                        }
                        #end diversity computation
                        
                        list(pred,div,SBname)
            } #end foreach 
  
      #Now get results 

      
      for (i in 1:length(folds2)){ #start get result out of predAndDiv
          
          yTEST <-  y[ -folds2[[i]]  ]


          
           if (tolower(object$eval.measure)=='spec') {

 
                      if (!is.null(predAndDiv[[i]][[1]]$predRBGA)) {

                        result$RBGA[i] <-  AUC::auc(specificity(predAndDiv[[i]][[1]]$predRBGA,yTEST))
                        predANDresp$RBGA[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predRBGA),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predDEOPT)){

                        result$DEOPT[i] <- AUC::auc(specificity(predAndDiv[[i]][[1]]$predDEOPT ,yTEST))
                        predANDresp$DEOPT[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predDEOPT),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predGENSA)) {

                        result$GENSA[i] <- AUC::auc(specificity( predAndDiv[[i]][[1]]$predGENSA,yTEST))
                        predANDresp$GENSA[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predGENSA),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predMALSCHAINS)) {

                        result$MALSCHAINS[i] <-  AUC::auc(specificity(predAndDiv[[i]][[1]]$predMALSCHAINS ,yTEST))
                        predANDresp$MALSCHAINS[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predMALSCHAINS),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predPSOPTIM)){

                        result$PSOPTIM[i] <-  AUC::auc(specificity( predAndDiv[[i]][[1]]$predPSOPTIM,yTEST))
                        predANDresp$PSOPTIM[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predPSOPTIM),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predSOMA)) {

                        result$SOMA[i] <-  AUC::auc(specificity( predAndDiv[[i]][[1]]$predSOMA,yTEST))
                        predANDresp$SOMA[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predSOMA),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predTABU)) {

                        result$TABU[i] <-  AUC::auc(specificity(predAndDiv[[i]][[1]]$predTABU ,yTEST))
                        predANDresp$TABU[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predTABU),response=yTEST)
                      }


                      if (!is.null(predAndDiv[[i]][[1]]$predLHNNLS)) {

                        result$LHNNLS[i] <-  AUC::auc(specificity(predAndDiv[[i]][[1]]$predLHNNLS ,yTEST))
                        predANDresp$LHNNLS[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predLHNNLS),response=yTEST)
                      }
                      if (!is.null(predAndDiv[[i]][[1]]$predGINNLS)) {

                        result$GINNLS[i] <- AUC::auc(specificity(predAndDiv[[i]][[1]]$predGINNLS ,yTEST))
                        predANDresp$GINNLS[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predGINNLS),response=yTEST)

                      }
                      if (!is.null(predAndDiv[[i]][[1]]$predNNloglik)) {

                        result$NNloglik[i] <-  AUC::auc(specificity(predAndDiv[[i]][[1]]$predNNloglik ,yTEST))
                        predANDresp$NNloglik[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predNNloglik),response=yTEST)
                      }





                      result$MEAN[i] <-  AUC::auc(specificity(predAndDiv[[i]][[1]]$predMEAN ,yTEST))
                      predANDresp$MEAN[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predMEAN),response=yTEST)



                      result$AUTHORITY[i] <-  AUC::auc(specificity(predAndDiv[[i]][[1]]$predAUTHORITY ,yTEST))
                      predANDresp$AUTHORITY[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predAUTHORITY),response=yTEST)



                      result$SB[i] <-  AUC::auc(specificity(predAndDiv[[i]][[1]]$predSB ,yTEST))
                      predANDresp$SB[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predSB),response=yTEST)


      } else if (tolower(object$eval.measure)=='sens') {

                      if (!is.null(predAndDiv[[i]][[1]]$predRBGA)) {

                        result$RBGA[i] <- AUC::auc(sensitivity(predAndDiv[[i]][[1]]$predRBGA ,yTEST))
                        predANDresp$RBGA[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predRBGA),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predDEOPT)){

                        result$DEOPT[i] <- AUC::auc(sensitivity(predAndDiv[[i]][[1]]$predDEOPT ,yTEST))
                        predANDresp$DEOPT[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predDEOPT),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predGENSA)) {

                        result$GENSA[i] <- AUC::auc(sensitivity(predAndDiv[[i]][[1]]$predGENSA ,yTEST))
                        predANDresp$GENSA[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predGENSA),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predMALSCHAINS)) {

                        result$MALSCHAINS[i] <- AUC::auc(sensitivity(predAndDiv[[i]][[1]]$predMALSCHAINS ,yTEST))
                        predANDresp$MALSCHAINS[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predMALSCHAINS),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predPSOPTIM)){

                        result$PSOPTIM[i] <- AUC::auc(sensitivity(predAndDiv[[i]][[1]]$predPSOPTIM ,yTEST))
                        predANDresp$PSOPTIM[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predPSOPTIM),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predSOMA)) {

                        result$SOMA[i] <-  AUC::auc(sensitivity(predAndDiv[[i]][[1]]$predSOMA ,yTEST))
                        predANDresp$SOMA[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predSOMA),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predTABU)) {

                        result$TABU[i] <-  AUC::auc(sensitivity(predAndDiv[[i]][[1]]$predTABU ,yTEST))
                        predANDresp$TABU[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predTABU),response=yTEST)
                      }


                      if (!is.null(predAndDiv[[i]][[1]]$predLHNNLS)) {

                        result$LHNNLS[i] <- AUC::auc(sensitivity(predAndDiv[[i]][[1]]$predLHNNLS ,yTEST))
                        predANDresp$LHNNLS[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predLHNNLS),response=yTEST)
                      }
                      if (!is.null(predAndDiv[[i]][[1]]$predGINNLS)) {

                        result$GINNLS[i] <-  AUC::auc(sensitivity(predAndDiv[[i]][[1]]$predGINNLS ,yTEST))
                        predANDresp$GINNLS[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predGINNLS),response=yTEST)

                      }
                      if (!is.null(predAndDiv[[i]][[1]]$predNNloglik)) {

                        result$NNloglik[i] <-  AUC::auc(sensitivity(predAndDiv[[i]][[1]]$predNNloglik ,yTEST))
                        predANDresp$NNloglik[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predNNloglik),response=yTEST)
                      }





                      result$MEAN[i] <-  AUC::auc(sensitivity(predAndDiv[[i]][[1]]$predMEAN ,yTEST))
                      predANDresp$MEAN[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predMEAN),response=yTEST)



                      result$AUTHORITY[i] <-  AUC::auc(sensitivity(predAndDiv[[i]][[1]]$predAUTHORITY ,yTEST))
                      predANDresp$AUTHORITY[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predAUTHORITY),response=yTEST)



                      result$SB[i] <-  AUC::auc(sensitivity(predAndDiv[[i]][[1]]$predSB ,yTEST))
                      predANDresp$SB[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predSB),response=yTEST)





      } else  if (tolower(object$eval.measure)=='auc') {



                      if (!is.null(predAndDiv[[i]][[1]]$predRBGA)) {
                        result$RBGA[i] <- AUC::auc(roc( predAndDiv[[i]][[1]]$predRBGA,yTEST))
                        predANDresp$RBGA[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predRBGA),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predDEOPT)){
                        result$DEOPT[i] <- AUC::auc(roc(predAndDiv[[i]][[1]]$predDEOPT,yTEST))
                        predANDresp$DEOPT[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predDEOPT),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predGENSA)) {
                        result$GENSA[i] <- AUC::auc(roc(predAndDiv[[i]][[1]]$predGENSA ,yTEST))
                        predANDresp$GENSA[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predGENSA),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predMALSCHAINS)) {
                        result$MALSCHAINS[i] <- AUC::auc(roc(predAndDiv[[i]][[1]]$predMALSCHAINS ,yTEST))
                        predANDresp$MALSCHAINS[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predMALSCHAINS),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predPSOPTIM)){
                        result$PSOPTIM[i] <- AUC::auc(roc(predAndDiv[[i]][[1]]$predPSOPTIM ,yTEST))
                        predANDresp$PSOPTIM[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predPSOPTIM),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predSOMA)) {
                        result$SOMA[i] <- AUC::auc(roc(predAndDiv[[i]][[1]]$predSOMA ,yTEST))
                        predANDresp$SOMA[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predSOMA),response=yTEST)
                      }

                      if (!is.null(predAndDiv[[i]][[1]]$predTABU)) {
                        result$TABU[i] <- AUC::auc(roc(predAndDiv[[i]][[1]]$predTABU ,yTEST))
                        predANDresp$TABU[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predTABU),response=yTEST)
                      }



                      if (!is.null(predAndDiv[[i]][[1]]$predLHNNLS)) {
                        result$LHNNLS[i] <- AUC::auc(roc(predAndDiv[[i]][[1]]$predLHNNLS ,yTEST))
                        predANDresp$LHNNLS[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predLHNNLS),response=yTEST)
                      }
                      if (!is.null(predAndDiv[[i]][[1]]$predGINNLS)) {
                        result$GINNLS[i] <- AUC::auc(roc(predAndDiv[[i]][[1]]$predGINNLS ,yTEST))
                        predANDresp$GINNLS[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predGINNLS),response=yTEST)

                      }
                      if (!is.null(predAndDiv[[i]][[1]]$predNNloglik)) {
                        result$NNloglik[i] <- AUC::auc(roc(predAndDiv[[i]][[1]]$predNNloglik ,yTEST))
                        predANDresp$NNloglik[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predNNloglik),response=yTEST)
                      }



                      result$MEAN[i] <- AUC::auc(roc(predAndDiv[[i]][[1]]$predMEAN ,yTEST))
                      predANDresp$MEAN[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predMEAN),response=yTEST)

                      result$AUTHORITY[i] <- AUC::auc(roc( predAndDiv[[i]][[1]]$predAUTHORITY,yTEST))
                      predANDresp$AUTHORITY[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predAUTHORITY),response=yTEST)

                      result$SB[i] <- AUC::auc(roc( predAndDiv[[i]][[1]]$predSB,yTEST))
                      predANDresp$SB[[i]] <- list(predicted=as.numeric(predAndDiv[[i]][[1]]$predSB),response=yTEST)

      }
    } #end get result out of predAndDiv
    
    for (i in 1:length(folds2)){
     divstore <- rbind(divstore, predAndDiv[[i]][[2]])
    }
      
    SBname <- numeric(10)  
    for (i in 1:length(folds2)){ 
      SBname[i] <- predAndDiv[[i]][[3]]
    }
      
} #end parallel
  
    if (diversity==TRUE){
        divstore <- aggregate(divstore[,2:ncol(divstore)], by=list(ensemble=divstore$ensemble),mean)
    } else {
        divstore <- NULL
    }

    final <- list()


    for (i in 1:length(result)) final[[i]] <- list(median=median(unlist(result[[i]])),
                                                   IQR=as.numeric(quantile(unlist(result[[i]]),probs=0.75)-quantile(unlist(result[[i]]),probs=0.25)),
                                                   folds=result[[i]],
                                                   predictions=predANDresp[[i]]
                                                  )
    names(final) <- names(result)
    final$SB$SBname <- perf
    final$SB$SBname <- SBname
    final$eval.measure <- object$eval.measure
    final$diversity <- divstore

  class(final) <- "CVhybridEnsemble"
  final
}
