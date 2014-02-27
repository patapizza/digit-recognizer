#!/usr/bin/Rscript

codebooks <- diag(10)

train <- function(path) {
    data <- read.csv(path)
    network <- list(b_input=rnorm(784), w_input_h1=matrix(rnorm(784 * 500), nrow=784, ncol=500), b_h1=rnorm(500), w_h1_output=matrix(rnorm(500 * 10), nrow=500, ncol=10))
    rprop(network, data)
}

rprop <- function(network, data, eta=0.1) {
    mse <- Inf
    repeat {
        sse <- 0
        for (i in 1:nrow(data)) {
            o <- fw_pass(network, data[i, ])
            error <- o[[3]] - codebooks[data[i, "label"] + 1, ]
            sse <- sse + sum(error ^ 2) / 2

            delta_h1_output <- error
            delta_input_h1 <- (network[["w_h1_output"]] * delta_h1_output) * o[[2]] * (1 - o[[2]])

            network[["w_input_h1"]] <- -eta * delta_input_h1 * o[[1]] + network[["w_input_h1"]]
            network[["w_h1_output"]] <- -eta * delta_h1_output * o[[2]] + network[["w_h1_output"]]
        }
        if (sse / nrow(data) >= mse)
            eta <- eta / 2
        mse <- sse / nrow(data)
        cat("MSE:", mse, "(", eta,")\n")
        if (mse < 0.01)
            break
    }
}

fw_pass <- function(network, input) {
    layer1 <- hidden_activations(input, network[["b_input"]], network[["w_input_h1"]])
    output <- hidden_activations(layer1, network[["b_h1"]], network[["w_h1_output"]])
    list(input, layer1, output)
}

sigmoid <- function(x) {
    1 / (1 + exp(-x))
}

hidden_activations <- function(o, b, w) {
    sigmoid(b + w * o)
}

train("./data/train_tiny.csv")
