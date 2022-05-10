context("Main functions: hybridEnsemble, predict and CVhybridEnsemble")



test_that("Test output hybridEnsemble and predict", {
skip_on_cran()
data(Credit)
hE <-hybridEnsemble(x=Credit[1:100,names(Credit) != 'Response'],
                    y=Credit$Response[1:100],
                    verbose=FALSE,
                    combine=c("rbga"),
                    RF.ntree=50,
                    AB.iter=50,
                    NN.size=5,
                    NN.decay=0,
                    SV.gamma = 2^-15,
                    SV.cost = 2^-5,
                    SV.degree=2,
                    SV.kernel='radial')
  
predictions <- predict(hE, newdata=Credit[1:100,names(Credit) != 'Response'])

expect_equal(class(hE),"hybridEnsemble")
expect_output(str(hE),"List of 35")
expect_output(str(predictions),"List of 5")

})



test_that("Test output CVhybridEnsemble", {
skip_on_cran()
data(Credit)
x <- Credit[1:200,names(Credit) != 'Response']
x <- x[,sapply(x,is.numeric)]
CVhE <- CVhybridEnsemble(x=x ,
                    y=Credit$Response[1:200],
                    verbose=FALSE,
                    diversity=TRUE,
                    filter=0.05,
                    KF.rp=1,
                    RF.ntree=50,
                    AB.iter=50,
                    NN.size=5,
                    NN.decay=0,
                    SV.gamma = 2^-15,
                    SV.cost = 2^-5,
                    SV.degree=2,
                    SV.kernel='radial')
expect_output(str(CVhE),"List of 4")
})


