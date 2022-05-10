#' Predict method for hybridEnsemble objects
#'
#' Prediction of new data using a hybridEnsemble model.
#' 
#' @param object An object of class hybridEnsemble created by the function  \code{hybridEnsemble}
#' @param newdata A data frame with the same predictors as in the training data
#' @param verbose TRUE or FALSE. Should information be printed to the screen
#' @param predict.all TRUE or FALSE. Should the predictions of all the members be returned?
#' @param ... Not currently used
#' 
#' @examples
#' 
#' data(Credit)
#' 
#' \dontrun{
#' hE <-hybridEnsemble(x=Credit[1:100,names(Credit) != 'Response'],
#'                     y=Credit$Response[1:100],
#'                     RF.ntree=50,
#'                     AB.iter=50,
#'                     NN.size=5,
#'                     NN.decay=0,
#'                     SV.gamma = 2^-15,
#'                     SV.cost = 2^-5,
#'                     SV.degree=2,
#'                     SV.kernel='radial')
#' 
#' predictions <- predict(hE, newdata=Credit[1:100,names(Credit) != 'Response'])
#' }
#' 
#' @references Ballings, M., Vercamer, D., Bogaert, M., Van den Poel, D.
#' @seealso \code{\link{hybridEnsemble}}, \code{\link{CVhybridEnsemble}}, \code{\link{importance.hybridEnsemble}}, \code{\link{plot.CVhybridEnsemble}}, \code{\link{summary.CVhybridEnsemble}}
#' @return A list containing the following vectors:
#' \item{predMEAN}{Predictions combined by the simple mean}
#' \item{SB}{A label denoting the single best algorithm: RF=Random Forest, LR= Bagged Logistic Regression, AB= AdaBoost, SV=Bagged Support Vector Machines, NN=Bagged Neural Networks, KF=Kernel Factory}
#' \item{predSB}{Predictions by the single best}
#' \item{predAUTHORITY}{Predictions combined by authority}
#' ..and all the combination methods that are requested in the \code{\link{hybridEnsemble}} function.
#' @author Michel Ballings, Dauwe Vercamer, Matthias Bogaert, and Dirk Van den Poel, Maintainer: \email{Michel.Ballings@@GMail.com}
#' @method predict hybridEnsemble
predict.hybridEnsemble <- function(object,newdata,verbose=FALSE, predict.all=FALSE, ...){
    
       predictions <- data.frame(matrix(NA,nrow=nrow(newdata),ncol=0))
       newdata <- newdata[,!object$constants]
       if (predict.all) baseclassifiers <- list()
       
      ########################################
      if (!is.null(object$LR)){ 
       #bagged logit
       predLR <- data.frame(matrix(nrow=nrow(newdata),ncol=length(object$LR)))
       for (i in 1:length(object$LR)) {
         predLR[,i] <- as.numeric(predict(object$LR[[i]],newx=data.matrix(newdata),type="response",s=object$LR.lambda))
       }
       if (predict.all) {
         baseclassifiers$predLR <- predLR
       }
       predictions$LR <- as.numeric(rowMeans(predLR))
       rm(predLR)
       if (is.null(object$calibratorLR)) {
          predictions$LR  <- rank(predictions$LR ,ties.method="min")/length(predictions$LR )
       } else {
          predictions$LR  <- .predict.calibrate(object=object$calibratorLR, newdata=predictions$LR )
       }
      }
      
       ######################################## 
       #random forest
      if (!is.null(object$RF)){
       if (predict.all) baseclassifiers$predRF <- data.frame(sapply(data.frame(predict(object$RF, newdata,type="prob", predict.all=TRUE)$individual,stringsAsFactors=FALSE),as.integer,simplify=FALSE))
  
       predictions$RF <- as.numeric(predict(object$RF, newdata,type="prob")[,2])

       #if one calibrator is NULL then all are NULL meaning that calibrate=FALSE in hybridEnsemble
       if (is.null(object$calibratorRF)) {
          predictions$RF <- rank(predictions$RF,ties.method="min")/length(predictions$RF)
       } else {
          predictions$RF <- .predict.calibrate(object=object$calibratorRF, newdata=predictions$RF)
       }
      }
       
      ########################################
      if (!is.null(object$AB)){
       #adaboost
       if (predict.all) baseclassifiers$predAB <- .predictada(object=object$AB,newdata)
       predictions$AB <- as.numeric(predict(object$AB, newdata,type="probs")[,2])
       
       if (is.null(object$calibratorAB)) {
          predictions$AB <- rank(predictions$AB,ties.method="min")/length(predictions$AB)
       } else {
          predictions$AB <- .predict.calibrate(object=object$calibratorAB, newdata=predictions$AB)
       }
      }
      
      ######################################## 
      if (!is.null(object$KF)){
       #kernel factory
       if (predict.all) baseclassifiers$predKF <- predict(object$KF, newdata, predict.all=TRUE)
       predictions$KF <- as.numeric(predict(object$KF, newdata))
       
       if (is.null(object$calibratorKF)) {
          predictions$KF <- rank(predictions$KF,ties.method="min")/length(predictions$KF)
       } else {
          predictions$KF <- .predict.calibrate(object=object$calibratorKF, newdata=predictions$KF)
       }
      }
     
       
      ######################################## 
     if (!is.null(object$NN)){
      #neural networks
       predNN <- data.frame(matrix(nrow=nrow(newdata),ncol=length(object$NN)))
       for (i in 1:length(object$NN)) {
           newdatascaled <- data.frame(sapply(newdata, as.numeric))
           newdatascaled <- data.frame(sapply(newdatascaled, function(x) if(length(unique(x))==2 && min(x)==1) x-1 else x))


           newdatascaled <- data.frame(t((t(newdatascaled) - ((object$minima[[i]] + object$maxima[[i]])/2))/((object$maxima[[i]]-object$minima[[i]])/2))) 
         
           predNN[,i] <- as.numeric(predict(object=object$NN[[i]],newdata=newdatascaled,type="raw"))  

       }
       if (predict.all)  baseclassifiers$predNN <- predNN
       
       predictions$NN <- as.numeric(rowMeans(predNN))
       rm(predNN)
       if (is.null(object$calibratorNN)) {
          predictions$NN <- rank(predictions$NN,ties.method="min")/length(predictions$NN)
       } else {
          predictions$NN <- .predict.calibrate(object=object$calibratorNN, newdata=predictions$NN)
       }
     }
      
      ########################################  
      if (!is.null(object$SV)){
       #support vector machines
       predSV <- data.frame(matrix(nrow=nrow(newdata),ncol=length(object$SV)))
       for (i in 1:length(object$SV)) {
           predSV[,i] <- as.numeric(attr(predict(object$SV[[i]],newdata, probability=TRUE),"probabilities")[,2])
           
       }      
       if (predict.all)  baseclassifiers$predSV <- predSV
       predictions$SV <- as.numeric(rowMeans(predSV))
       rm(predSV)
       
       if (is.null(object$calibratorSV)) {
          predictions$SV <- rank(predictions$SV,ties.method="min")/length(predictions$SV)
       } else {
          predictions$SV <- .predict.calibrate(object=object$calibratorSV, newdata=predictions$SV)
       }
      }
       
      ########################################
      if (!is.null(object$RoF)){
       #rotation forest
       if (predict.all)  baseclassifiers$predRoF <- data.frame(predict(object$RoF,newdata,all=TRUE))
       
       predictions$RoF <- as.numeric(predict(object$RoF,newdata[,sapply(newdata,is.numeric)]))
       
       
       if (is.null(object$calibratorRoF)) {
          predictions$RoF <- rank(predictions$RoF,ties.method="min")/length(predictions$RoF)
       } else {
          predictions$RoF <- .predict.calibrate(object=object$calibratorRoF, newdata=predictions$RoF)
       }  
      }
       
      ######################################## 
      if (!is.null(object$x_KN)){
       #k-nearest neighbors
  

       newdata_KN <- data.frame(sapply(newdata, as.numeric))
       newdata_KN <- data.frame(sapply(newdata_KN, function(x) if(length(unique(x))==2 && min(x)==1) x-1 else x))
       newdata_KN <- data.frame(t((t(newdata_KN)-object$minimaKN)/(object$maximaKN-object$minimaKN)))
          
       predKN <- data.frame(matrix(nrow=nrow(newdata_KN),ncol=object$KN.size))
       for (i in 1:object$KN.size){
          ind <- sample(1:nrow(object$x_KN),size=round(nrow(object$x_KN)), replace=TRUE)
          #retrieve the indicators of the k nearest neighbors of the query data 
          indicatorsKN <- as.integer(knnx.index(data=object$x_KN[ind,], query=newdata_KN, k=object$KN.K))
          #retrieve the actual y from the tarining set
          predKNoptimal <- as.integer(as.character(object$y_KN[indicatorsKN]))
          #if k > 1 than we take the proportion of 1s
          predKN[,i] <- rowMeans(data.frame(matrix(data=predKNoptimal,ncol=object$KN.K,nrow=nrow(newdata_KN))))
       }
       if (predict.all)  baseclassifiers$predKN <- predKN
       predictions$KN <- rowMeans(predKN)
       rm(predKN)
  
       if (is.null(object$calibratorKN)) {
          predictions$KN <- rank(predictions$KN,ties.method="min")/length(predictions$KN)
       } else {
          predictions$KN <- .predict.calibrate(object=object$calibratorKN, newdata=predictions$KN)
       }  
      }
       
      ######################################## 
      if (!is.null(object$NB)){
       #bagged naive bayes
       
       #Use ame data as KN
       if (!exists("newdata_KN")){
          newdata_KN <- data.frame(sapply(newdata, as.numeric))
          newdata_KN <- data.frame(sapply(newdata_KN, function(x) if(length(unique(x))==2 && min(x)==1) x-1 else x))
          newdata_KN <- data.frame(t((t(newdata_KN)-object$minimaKN)/(object$maximaKN-object$minimaKN)))
       }
       
       
       predNB <- data.frame(matrix(nrow=nrow(newdata),ncol=length(object$NB)))
       for (i in 1:length(object$NB)) {
         predNB[,i] <- predict(object=object$NB[[i]], newdata_KN, type = "raw", threshold = 0.001)[,2]
       }
       if (predict.all) {
         baseclassifiers$predNB <- predNB
       }
       predictions$NB <- as.numeric(rowMeans(predNB))
       rm(predNB)
       
       if (is.null(object$calibratorNB)) {
          predictions$NB <- rank(predictions$NB,ties.method="min")/length(predictions$NB)
       } else {
          predictions$NB <- .predict.calibrate(object=object$calibratorNB, newdata=predictions$NB)
       }
      }
  
       #####     
  
       result <- list()

       if (tolower('rbga') %in% tolower(object$combine)) {
         if (verbose==TRUE) cat('   Genetic Algorithm \n') 
         result$predRBGA <- as.numeric(rowSums(t(t(predictions)*as.numeric(object$weightsRBGA))))
       }
       if (tolower('DEopt') %in% tolower(object$combine)) {
         if (verbose==TRUE) cat('   Differential Evolutionary Algorithm \n')     
         result$predDEOPT <- as.numeric(rowSums(t(t(predictions)*as.numeric(object$weightsDEOPT))))
       }
       if (tolower('GenSA') %in% tolower(object$combine)) {
         if (verbose==TRUE) cat('   Generalized Simulated Annealing \n')     
         result$predGENSA <- as.numeric(rowSums(t(t(predictions)*as.numeric(object$weightsGENSA))))
       }
       if (tolower('malschains') %in% tolower(object$combine)) {
         if (verbose==TRUE) cat('   Memetic Algorithm with Local Search Chains \n')   
         result$predMALSCHAINS <- as.numeric(rowSums(t(t(predictions)*as.numeric(object$weightsMALSCHAINS))))
       }
       if (tolower('psoptim') %in% tolower(object$combine)) {
         if (verbose==TRUE) cat('   Particle Swarm Optimization \n')   
         result$predPSOPTIM <- as.numeric(rowSums(t(t(predictions)*as.numeric(object$weightsPSOPTIM))))
       }
       if (tolower('soma') %in% tolower(object$combine)) {
         if (verbose==TRUE) cat('   Self-Organising Migrating Algorithm \n') 
         result$predSOMA <- as.numeric(rowSums(t(t(predictions)*as.numeric(object$weightsSOMA))))
       }
       if (tolower('tabu') %in% tolower(object$combine)) {
         if (verbose==TRUE) cat('   Tabu Search Algorithm \n')   
         result$predTABU <- as.numeric(rowSums(t(t(predictions)*as.numeric(object$weightsTABU))))
       }
       
       if (tolower('LHNNLS') %in% tolower(object$combine)) {
         if (verbose==TRUE) cat('   Lawson-Hanson Non-negative least squares \n')    
         result$predLHNNLS <- as.numeric(crossprod(t(predictions), object$weightsLHNNLS))
       }
       
       if (tolower('GINNLS') %in% tolower(object$combine)) {
        if (verbose==TRUE) cat('   Goldfarb-Idnani Non-negative least squares \n')
        result$predGINNLS <- as.numeric(crossprod(t(predictions), object$weightsGINNLS))
       }
       
       if (tolower('NNloglik') %in% tolower(object$combine)) {
         if (verbose==TRUE) cat('   Non-negative binomial likelihood  \n')
         trimLogit <- function(x, trim=0.00001) {
           x[x < trim] <- trim
           x[x > (1-trim)] <- (1-trim)
           foo <- log(x/(1-x))
           return(foo)
         }
         
         result$predNNloglik <- as.numeric(plogis(crossprod(t(trimLogit(predictions)), object$weightsNNloglik)))
       }


       #################################################
       
       if (verbose==TRUE) cat('   Mean \n')
       result$predMEAN <- as.numeric(rowMeans(predictions))
       
       if (verbose==TRUE) cat('   Single Best \n')
       result$SB <- object$SB
       result$predSB <- predictions[,colnames(predictions) %in% object$SB]
       
       if (verbose==TRUE) cat('   Authority \n')
       result$predAUTHORITY <- as.numeric(rowSums(t(t(predictions)*as.numeric(object$weightsAUTHORITY))))
    
       if (predict.all) {
         result$subensembles <- predictions
         result$baseclassifiers <- baseclassifiers
       }
         
    return(result)
}




