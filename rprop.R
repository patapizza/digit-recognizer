#!/usr/bin/Rscript

codebooks <- diag(10)

train <- function(path) {
    data <- read.csv(path)
    network <- list(b_input=vector(), w_input_h1=matrix(vector(), nrow=784, ncol=500), b_h1=vector(), w_h1_output=matrix(vector(), nrow=500, ncol=10))
    rprop(network, data)
}

rprop <- function(network, data) {
    repeat {
        sse <- 0
        for (i in 1:nrow(data)) {
            o <- fw_pass(data[i, ])
            error <- (codebooks[data[i, 1], ] - o[[length(o)]]) ^ 2
            sse <- sse + error
        }
        if (sse / nrow(data) < 0.01)
            break
    }
}

fw_pass <- function(input) {
    layer1 <- hidden_activations(input)
    output <- hidden_activations(layer1)
    return list(input, layer1, output)
}

sigmoid <- function(x) {
    return 1 / (1 + exp(-x))
}

hidden_activations <- function(o, b, w) {
    return sigmoid(b + w * o)
}
