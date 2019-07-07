library(data.table)
dttrn <- fread("~/Documents/Personal/recursion/data/train.csv")
dttst <- fread("~/Documents/Personal/recursion/data/test.csv")
dttrn[,.N,by = c('sirna', 'plate')][order(N)]
dtfolds = dttrn[,.N,'experiment']
dtfolds$fold = (1:nrow(dtfolds))%%5
dtfolds = dtfolds[,c('experiment', 'fold')]
fwrite(dtfolds, "~/Documents/Personal/recursion/data/folds.csv")
