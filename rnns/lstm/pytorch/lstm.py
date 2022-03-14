import torch
import torch.nn as nn


# first, define our model and initialize the learnable weights and biases (parameters):
class LSTM(nn.Module):
    def __init__(self, dims, hidden_dims, learn_h0=False, learn_c0=False,
                 activation_h=nn.Tanh,
                 activation_o=nn.Sigmoid, activation_f=nn.Tanh,
                 activation_i=nn.Sigmoid, activation_j=nn.Sigmoid,
                 rnn_mode=True):
        super().__init__()
        # it is fine to hard code these 
        self.activation_h = activation_h()
        self.activation_o = activation_o()
        self.activation_f = activation_f()
        self.activation_i = activation_i()
        self.activation_j = activation_j()

        # parameters of the (recurrent) hidden layer
        self.W_o = nn.Parameter(torch.randn(dims, hidden_dims) * .1)
        self.b_o = nn.Parameter(torch.zeros(1, hidden_dims))
        self.W_f = nn.Parameter(torch.randn(dims, hidden_dims) * .1)
        self.b_f = nn.Parameter(torch.zeros(1, hidden_dims))
        self.W_i = nn.Parameter(torch.randn(dims, hidden_dims) * .1)
        self.b_i = nn.Parameter(torch.zeros(1, hidden_dims))
        self.W_j = nn.Parameter(torch.randn(dims, hidden_dims) * .1)
        self.b_j = nn.Parameter(torch.zeros(1, hidden_dims))

        self.U_o = nn.Parameter(
            torch.randn(hidden_dims, hidden_dims) * .1
        )
        self.U_f = nn.Parameter(
            torch.randn(hidden_dims, hidden_dims) * .1
        )
        self.U_i = nn.Parameter(
            torch.randn(hidden_dims, hidden_dims) * .1
        )
        self.U_j = nn.Parameter(
            torch.randn(hidden_dims, hidden_dims) * .1
        )

        if not rnn_mode:
            self.U_o.zero_()
            self.U_f.zero_()
            self.U_i.zero_()
            self.U_j.zero_()
            self.U_o.requires_grad = False
            self.U_f.requires_grad = False
            self.U_i.requires_grad = False
            self.U_j.requires_grad = False

        # initial hidden state
        self.h_0 = nn.Parameter(
            torch.zeros(1, hidden_dims),
            requires_grad=learn_h0  # only train this if enabled
        )

        # initial cell state
        self.c_0 = nn.Parameter(
            torch.zeros(1, hidden_dims),
            requires_grad=learn_c0  # only train this if enabled
        )

        # output layer (fully connected)
        self.W_y = nn.Parameter(torch.randn(hidden_dims, dims) * .1)
        self.b_y = nn.Parameter(torch.zeros(1, dims))

    def step(self, x_t, h, c):
        #  forward pass for a single time step
        # hint: a more clever implementation could combine all these and select the different parts later
        j = self.activation_j(
            torch.matmul(x_t, self.W_j) + torch.matmul(h, self.U_j) + self.b_j
        )
        i = self.activation_i(
            torch.matmul(x_t, self.W_i) + torch.matmul(h, self.U_i) + self.b_i
        )
        f = self.activation_f(
            torch.matmul(x_t, self.W_f) + torch.matmul(h, self.U_f) + self.b_f
        )
        o = self.activation_o(
            torch.matmul(x_t, self.W_o) + torch.matmul(h, self.U_o) + self.b_o
        )

        c = f * c + i * j

        h = o * self.activation_h(c)

        return h, c  # returning new hidden and cell state

    def iterate_series(self, x, h, c):
        # apply rnn to each time step and give an output (many-to-many task)
        batch_size, n_steps, dimensions = x.shape

        # can use cell states list here but only the last cell is required
        hidden_states = []
        # iterate over time axis (1)
        for t in range(n_steps):
            # give previous hidden state and input from the current time step
            h, c = self.step(x[:, t], h, c)
            hidden_states.append(h)
        hidden_states = torch.stack(hidden_states, 1)

        # fully connected output
        y_hat = hidden_states.reshape(batch_size * n_steps, -1)  # flatten steps and batch size (bs * )
        y_hat = torch.matmul(y_hat, self.W_y) + self.b_y
        y_hat = y_hat.reshape(batch_size, n_steps, -1)  # regains structure
        return y_hat, hidden_states[:, -1], c

    def forward(self, x, h, c):
        # x: b, t, d
        batch_size = x.shape[0]
        if h is None:
            h = self.h_0.repeat_interleave(batch_size, 0)
        if c is None:
            c = self.c_0.repeat_interleave(batch_size, 0)
        y_hat, h, c = self.iterate_series(x, h, c)
        return y_hat, h, c

    def auto_regression(self, x_0, h, c, steps):
        # one-to-many task (steps = \delta')
        x_prev = x_0
        y_hat = []
        # iterate over time axis (1)
        for t in range(steps):
            # give previous hidden state and input from the current time step
            h, c = self.step(x_prev, h, c)
            # here we need to apply the output layer on each step individually
            x_prev = torch.matmul(h, self.W_y) + self.b_y

            y_hat.append(x_prev)
        y_hat = torch.stack(y_hat, 1)
        return y_hat, h, c

    def many_to_one(self, x, h, c):
        # not required
        # returns the last prediction and the hidden state
        y_hat, h, c = self(x, h, c)
        return y_hat[:, -1], h, c  # only return the last prediction

    def many_to_many_async(self, x, h, c, steps):
        # not required
        # combines many-to-one and one-to-many
        x_0, h, c = self.many_to_one(x, h, c)
        return self.auto_regression(x_0, h, c, steps)
