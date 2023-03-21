# Reading file & extracting variables from first line
SDM_data <- readLines("SDM_3.dat")
r <- as.numeric(strsplit(SDM_data[1], split=" ")[[1]][1])
B <- as.numeric(strsplit(SDM_data[1], split=" ")[[1]][2])
K <- as.numeric(strsplit(SDM_data[1], split=" ")[[1]][3])
nf <- as.numeric(strsplit(SDM_data[1], split=" ")[[1]][4])

# SDM_data[2] is a blank line so we start from SDM_data[3]
freqs <- c()
for (j in 3:length(SDM_data)){
  ifelse(SDM_data[j] == "", break, freqs <- c(freqs, as.numeric(SDM_data[j])))
}

# We can then construct a list, storing matrix parts for S_{ij}
S_list <- list()
# We start the iteration from j+1 (we had a blank line for j from previous loop)
for (k in seq((j+1), length(SDM_data), by=2052)){
  S_list[[length(S_list)+1]] <- list(SDM_data[k],
                                     as.numeric(SDM_data[(k+1):(k+1025)]) +
                                       1i*as.numeric(SDM_data[(k+1027):(k+2051)]))
}