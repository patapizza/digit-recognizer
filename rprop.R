#!/usr/bin/Rscript

codebooks <- diag(10)

train <- function(path) {
    data <- read.csv(path)

    # rescaling data into [0,1] interval
    data <- cbind(data["label"], data[, 2:ncol(data)] / 255)

    minibatches <- mk_minibatches(data)
    # to test

    network <- list(b_input=rnorm(784), w_input_h1=matrix(rnorm(784 * 500), nrow=784, ncol=500), b_h1=rnorm(500), w_h1_output=matrix(rnorm(500 * 10), nrow=500, ncol=10))
    rprop(network, data)
}

mk_minibatches <- function(data, samples_per_digit=2) {
    min <- min(nrow(data[data$label == 0,]),
               nrow(data[data$label == 1,]),
               nrow(data[data$label == 2,]),
               nrow(data[data$label == 3,]),
               nrow(data[data$label == 4,]),
               nrow(data[data$label == 5,]),
               nrow(data[data$label == 6,]),
               nrow(data[data$label == 7,]),
               nrow(data[data$label == 8,]),
               nrow(data[data$label == 9,]))
    samples_per_digit <- if (min < samples_per_digit) min else samples_per_digit
    minibatches <- list()
    for (i in 0:floor(min / samples_per_digit))
        minibatches <- rbind(minibatches, rbind(data[data$label == 0,][i * samples_per_digit + 1:i * samples_per_digit + samples_per_digit,],
                                                data[data$label == 1,][i * samples_per_digit + 1:i * samples_per_digit + samples_per_digit,],
                                                data[data$label == 2,][i * samples_per_digit + 1:i * samples_per_digit + samples_per_digit,],
                                                data[data$label == 3,][i * samples_per_digit + 1:i * samples_per_digit + samples_per_digit,],
                                                data[data$label == 4,][i * samples_per_digit + 1:i * samples_per_digit + samples_per_digit,],
                                                data[data$label == 5,][i * samples_per_digit + 1:i * samples_per_digit + samples_per_digit,],
                                                data[data$label == 6,][i * samples_per_digit + 1:i * samples_per_digit + samples_per_digit,],
                                                data[data$label == 7,][i * samples_per_digit + 1:i * samples_per_digit + samples_per_digit,],
                                                data[data$label == 8,][i * samples_per_digit + 1:i * samples_per_digit + samples_per_digit,],
                                                data[data$label == 9,][i * samples_per_digit + 1:i * samples_per_digit + samples_per_digit,]))
    minibatches
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

logistic <- function(x) {
    1 / (1 + exp(-x))
}

hidden_activations <- function(o, b, w) {
    logistic(b + w * o)
}

train("./data/train_tiny.csv")
