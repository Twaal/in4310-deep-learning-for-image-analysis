import pytest
import torch

from model import LSTMCell

HIDDEN_STATE_SIZE, INPUT_SIZE = 7, 3
LSTM = LSTMCell(hidden_state_size=HIDDEN_STATE_SIZE, input_size=INPUT_SIZE)


def turn_off_all_the_gates():
    # Turn off all the gates by setting all the weights to 0 and biases to a very low -ve value
    # Gates will be off because sigmoid of low -ve value is 0
    for name, w in LSTM.named_parameters():
        if 'weight' in name:
            w.data = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(0.))
        if 'bias' in name:
            w.data = torch.nn.Parameter(-1000. * torch.ones(1, HIDDEN_STATE_SIZE))


@pytest.mark.parametrize('x, hidden_state, expected_cell_state', [
    # input, hidden state, memory are all 0
    ([0] * INPUT_SIZE, [0] * (2 * HIDDEN_STATE_SIZE), torch.tensor([0.] * HIDDEN_STATE_SIZE)),
    # input = 0, hidden_state = 1 old_memory = 0
    ([0] * INPUT_SIZE, [1] * HIDDEN_STATE_SIZE + [0] * HIDDEN_STATE_SIZE, torch.tensor([1.] * HIDDEN_STATE_SIZE)),
    # input = 0, hidden_state = 1 old_memory = 1
    ([0] * INPUT_SIZE, [1] * HIDDEN_STATE_SIZE + [1] * HIDDEN_STATE_SIZE, torch.tensor([1.] * HIDDEN_STATE_SIZE)),
    # input = 1, hidden_state = 0 old_memory = 0
    ([1] * INPUT_SIZE, [0] * (2 * HIDDEN_STATE_SIZE), torch.tanh(INPUT_SIZE * torch.ones(HIDDEN_STATE_SIZE))),
    # input = 1, hidden_state = 1 old_memory = 0
    ([1] * INPUT_SIZE, [1] * HIDDEN_STATE_SIZE + [0] * HIDDEN_STATE_SIZE,
     torch.tanh((INPUT_SIZE + HIDDEN_STATE_SIZE) * torch.ones(HIDDEN_STATE_SIZE))),
    # input = 1, hidden_state = 1 old_memory = 1
    ([1] * INPUT_SIZE, [1] * HIDDEN_STATE_SIZE + [1] * HIDDEN_STATE_SIZE,
     torch.tanh((INPUT_SIZE + HIDDEN_STATE_SIZE) * torch.ones(HIDDEN_STATE_SIZE)))
])
def test_lstm_input_gate_and_memory_cell_bias_zero(x, hidden_state, expected_cell_state):
    """
    This test tries to test only the input gate equation and the new memory equation.
    It's not possible to isolate an GRU equation without having a separate instance variable for each equation.
    Therefore, this test will be able to test those 2 equations in isolation only if everything else has been correctly
    implemented.

    - The basic idea is to turn off all the gates except the input date. This can be done by setting weights or biases
      or both to a low -ve value so that the sigmoid is 0 (assuming that the input is a low +ve value which we ensure).
    - This test sets weights to low -ve value and biases to 0.
    - The input gate weights are set to a high +ve value to turn it on (for +ve valued input > 1).
    - The memory cell weights are set to 1 so that the logit is simply the provided input.
    - Once that has been done, we can pass different inputs and hidden states to the forward function and match the
      result against the expected values.

    :param x: The input to the GRU
    :param hidden_state: The hidden state to be provided as the input to the GRU
    :param expected_cell_state: The expected correct output
    """
    # Convert inputs to tensors with an extra dimension added for batch size
    x = torch.tensor(x).unsqueeze(0).type(torch.float)
    hidden_state = torch.tensor(hidden_state).unsqueeze(0).type(torch.float)
    # Turn of all the gates of GRU
    turn_off_all_the_gates()

    # Set input gate's and memory cell's biases to 0
    LSTM.bias_i.data = torch.nn.Parameter(0. * torch.ones(1, HIDDEN_STATE_SIZE))
    LSTM.bias.data = torch.nn.Parameter(0. * torch.ones(1, HIDDEN_STATE_SIZE))
    # Now turn on the input gate and memory update via weights
    LSTM.weight_i.data = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(1000.))
    LSTM.weight.data = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(1.))

    # Test the inputs
    cell_state = LSTM(x, hidden_state.clone())[0, HIDDEN_STATE_SIZE:]

    assert torch.isclose(expected_cell_state, cell_state).all(),\
        'Input gate and memory update test failed when open via weights'

    # Now turn on the input gate via biases instead
    LSTM.bias_i.data = torch.nn.Parameter(1000. * torch.ones(1, HIDDEN_STATE_SIZE))
    LSTM.weight_i.data = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(0.))

    # Test the inputs
    cell_state = LSTM(x, hidden_state.clone())[0, HIDDEN_STATE_SIZE:]

    assert torch.isclose(expected_cell_state, cell_state).all(),\
        'Input gate and memory update test failed when open via biases'

    # Now turn on the input gate only partially via biases
    LSTM.bias_i.data = torch.nn.Parameter(0.5 * torch.ones(1, HIDDEN_STATE_SIZE))

    # Get the cell states
    cell_state = LSTM(x, hidden_state.clone())[0, HIDDEN_STATE_SIZE:]
    # Update the expected values as the gate is now only partially open
    expected_cell_state = expected_cell_state * torch.sigmoid(torch.tensor(0.5))

    assert torch.isclose(expected_cell_state, cell_state).all(), \
        'Input gate and memory update test failed when open via biases'


@pytest.mark.parametrize('x, hidden_state, bias', [
    # Input values do not matter for this test. The value of bias is what is important.
    # Note that the bias values are small because tanh saturates and is 1 for high +ve values.
    ([0] * INPUT_SIZE, [0] * (2 * HIDDEN_STATE_SIZE), 0.1),
    ([0] * INPUT_SIZE, [1] * HIDDEN_STATE_SIZE + [0] * HIDDEN_STATE_SIZE, 0.2),
    ([0] * INPUT_SIZE, [1] * HIDDEN_STATE_SIZE + [1] * HIDDEN_STATE_SIZE, 0.3),
    ([1] * INPUT_SIZE, [0] * (2 * HIDDEN_STATE_SIZE), 0.4),
    ([1] * INPUT_SIZE, [1] * HIDDEN_STATE_SIZE + [0] * HIDDEN_STATE_SIZE, 0.5),
    ([1] * INPUT_SIZE, [1] * HIDDEN_STATE_SIZE + [1] * HIDDEN_STATE_SIZE, 0.6),
    ([3] * INPUT_SIZE, [7] * HIDDEN_STATE_SIZE + [11] * HIDDEN_STATE_SIZE, 0.7),
    ([4] * INPUT_SIZE, [8] * HIDDEN_STATE_SIZE + [12] * HIDDEN_STATE_SIZE, 0.8),
    ([5] * INPUT_SIZE, [9] * HIDDEN_STATE_SIZE + [13] * HIDDEN_STATE_SIZE, 0.9),
    ([6] * INPUT_SIZE, [10] * HIDDEN_STATE_SIZE + [14] * HIDDEN_STATE_SIZE, 1.0),
])
def test_lstm_input_gate_and_memory_cell_bias_non_zero(x, hidden_state, bias):
    """
    This test tries to test only the input gate equation and the new memory equation.
    It's not possible to isolate an GRU equation without having a separate instance variable for each equation.
    Therefore this test will be able to test those 2 equations in isolation only if everything else has been correctly
    implemented.

    - The basic idea is to turn off all the gates except the input date. This can be done by setting weights or biases
      or both to a low -ve value so that the sigmoid is 0 (assuming that the input is a low +ve value which we ensure).
    - This test sets weights to 0 and biases to a low -ve value.
    - The input gate biases are set to a high +ve value to turn it on (for +ve valued input > 1).
    - The memory cell biases are set to the value provided via argument.
    - Once that has been done, we can pass different inputs and hidden states to the forward function and match the
      result against the expected values.


    :param x: The input to the GRU
    :param hidden_state: The hidden state to be provided as the input to the GRU
    :param bias: The bias value of the memory cell equation
    """
    # Convert inputs to tensors with an extra dimension added for batch size
    x = torch.tensor(x).unsqueeze(0).type(torch.float)
    hidden_state = torch.tensor(hidden_state).unsqueeze(0).type(torch.float)
    # Turn of all the gates of GRU
    turn_off_all_the_gates()

    # Now turn on the input gate by setting bias to a high +ve value
    LSTM.bias_i.data = torch.nn.Parameter(1000. * torch.ones(1, HIDDEN_STATE_SIZE))
    # Set the bias to the value provided as an argument
    LSTM.bias.data = torch.nn.Parameter(bias * torch.ones(1, HIDDEN_STATE_SIZE))
    # Test the inputs
    cell_state = LSTM(x, hidden_state)[0, HIDDEN_STATE_SIZE:]
    # Since the weights are 0 and we're ensuring that input gate is on, cell state will be updated only via bias
    expected_cell_state = torch.tanh(bias * torch.ones(1, HIDDEN_STATE_SIZE))

    assert torch.isclose(expected_cell_state, cell_state).all()


@pytest.mark.parametrize('x, hidden_state', [
    # The input and the hidden state should not affect the output.
    # Only tThe cell state matters here. The input cell sate provided here should also be the output
    ([1] * INPUT_SIZE, [7.] * HIDDEN_STATE_SIZE + [0] * HIDDEN_STATE_SIZE),
    ([2] * INPUT_SIZE, [8.] * HIDDEN_STATE_SIZE + [0.1] * HIDDEN_STATE_SIZE),
    ([3] * INPUT_SIZE, [9.] * HIDDEN_STATE_SIZE + [0.5] * HIDDEN_STATE_SIZE),
    ([4] * INPUT_SIZE, [10.] * HIDDEN_STATE_SIZE + [1.5] * HIDDEN_STATE_SIZE),
    ([5] * INPUT_SIZE, [11.] * HIDDEN_STATE_SIZE + [2] * HIDDEN_STATE_SIZE),
    ([6] * INPUT_SIZE, [12.] * HIDDEN_STATE_SIZE + [42] * HIDDEN_STATE_SIZE)
])
def test_lstm_forget_gate(x, hidden_state):
    """
    This test tries to test only the forget gate equation by turning it on (it's off in other tests already).
    It's not possible to isolate an GRU equation without having a separate instance variable for each equation.
    Therefore this test will be able to test forget gate in isolation only if everything else has been correctly
    implemented.

    - The basic idea is to turn off all the gates except the forget date. This can be done by setting weights or biases
      or both to a low -ve value so that the sigmoid is 0 (assuming that the input is a low +ve value which we ensure).
    - This test turns on forget gate in 2 different ways: via weights and via bias.
    - Once that has been done, inputs and hidden states are passed to the forward function. The resulting cell state
      is matched against the expected cell state which in this case is the old cell state.


    :param x: The input to the GRU
    :param hidden_state: The hidden state to be provided as the input to the GRU
    """
    # Convert inputs to tensors with an extra dimension added for batch size
    x = torch.tensor(x).unsqueeze(0).type(torch.float)
    hidden_state = torch.tensor(hidden_state).unsqueeze(0).type(torch.float)
    # Turn of all the gates of GRU
    turn_off_all_the_gates()

    # Set forget gate's biases to 0
    LSTM.bias_f.data = torch.nn.Parameter(0. * torch.ones(1, HIDDEN_STATE_SIZE))
    # Now turn on the forget gate via weights
    LSTM.weight_f.data = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(1000.))

    # Test the inputs
    cell_state = LSTM(x, hidden_state.clone())[0, HIDDEN_STATE_SIZE:]
    # We expect the input cell state to also be the output cell state
    # as everything is 0 except the forget gate which is 1
    expected_cell_state = hidden_state[0, HIDDEN_STATE_SIZE:]

    assert torch.isclose(expected_cell_state, cell_state).all(), 'Forget gate test failed with its bias set to 0.'

    # Now turn on the forget gate via biases instead
    LSTM.bias_f.data = torch.nn.Parameter(1000. * torch.ones(1, HIDDEN_STATE_SIZE))
    LSTM.weight_f.data = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(0.))

    # Test the inputs
    cell_state = LSTM(x, hidden_state.clone())[0, HIDDEN_STATE_SIZE:]
    # We expect the input cell state to also be the output cell state
    # as everything is 0 except the forget gate which is 1
    expected_cell_state = hidden_state[0, HIDDEN_STATE_SIZE:]

    assert torch.isclose(expected_cell_state, cell_state).all(), 'Forget gate test failed with its weights set to 0.'

    # Now turn on the forget gate only partially via biases
    partial_bias_value = 0.5
    LSTM.bias_f.data = torch.nn.Parameter(partial_bias_value * torch.ones(1, HIDDEN_STATE_SIZE))

    # Test the inputs
    cell_state = LSTM(x, hidden_state.clone())[0, HIDDEN_STATE_SIZE:]
    # We expect the input cell state to also be the output cell state
    # as everything is 0 except the forget gate which is 1
    expected_cell_state = hidden_state[0, HIDDEN_STATE_SIZE:] * torch.sigmoid(torch.tensor(partial_bias_value))

    assert torch.isclose(expected_cell_state, cell_state).all(), 'Forget gate test failed when partially on.'


@pytest.mark.parametrize('x, hidden_state', [
    # The input and the hidden state should not affect the output.
    # Only the cell state matters here.
    ([1] * INPUT_SIZE, [7.] * HIDDEN_STATE_SIZE + [0] * HIDDEN_STATE_SIZE),
    ([2] * INPUT_SIZE, [8.] * HIDDEN_STATE_SIZE + [0.1] * HIDDEN_STATE_SIZE),
    ([3] * INPUT_SIZE, [9.] * HIDDEN_STATE_SIZE + [0.5] * HIDDEN_STATE_SIZE),
    ([4] * INPUT_SIZE, [10.] * HIDDEN_STATE_SIZE + [1.5] * HIDDEN_STATE_SIZE),
    ([5] * INPUT_SIZE, [11.] * HIDDEN_STATE_SIZE + [2] * HIDDEN_STATE_SIZE),
    ([6] * INPUT_SIZE, [12.] * HIDDEN_STATE_SIZE + [42] * HIDDEN_STATE_SIZE)
])
def test_lstm_output_gate_and_hidden_state(x, hidden_state):
    """
    This test tries to test only the forget gate equation by turning it on (it's off in other tests already).
    It's not possible to isolate an GRU equation without having a separate instance variable for each equation.
    Therefore this test will be able to test forget gate in isolation only if everything else has been correctly
    implemented.

    - The basic idea is to turn off all the gates except the forget date. This can be done by setting weights or biases
      or both to a low -ve value so that the sigmoid is 0 (assuming that the input is a low +ve value which we ensure).
    - This test turns on forget gate in 2 different ways: via weights and via bias.
    - Once that has been done, inputs and hidden states are passed to the forward function. The resulting cell state
      is matched against the expected cell state which in this case is the old cell state.


    :param x: The input to the GRU
    :param hidden_state: The hidden state to be provided as the input to the GRU
    """
    # Convert inputs to tensors with an extra dimension added for batch size
    x = torch.tensor(x).unsqueeze(0).type(torch.float)
    hidden_state = torch.tensor(hidden_state).unsqueeze(0).type(torch.float)
    # Turn of all the gates of GRU
    turn_off_all_the_gates()

    # Set output gate's biases to 0
    LSTM.bias_o.data = torch.nn.Parameter(0. * torch.ones(1, HIDDEN_STATE_SIZE))
    # Turn on the output gate via weights
    LSTM.weight_o.data = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(1000.))
    # We also need to turn on the forget gate because output gate can only be tested via the new hidden state values
    # which will always be 0 if either of the forget gate or input gate is not on.
    LSTM.bias_f.data = torch.nn.Parameter(1000. * torch.ones(1, HIDDEN_STATE_SIZE))

    # Test the inputs
    new_hidden_state = LSTM(x, hidden_state)[0, :HIDDEN_STATE_SIZE]
    # We expect the input cell state to also be the output cell state
    # So the output hidden state should simply be tanh(input cell state)
    expected_hidden_state = torch.tanh(hidden_state[0, HIDDEN_STATE_SIZE:])

    assert torch.isclose(expected_hidden_state, new_hidden_state).all(),\
        'Output gate test failed with its bias set to 0.'

    # Now turn on the output gate via biases instead
    LSTM.bias_o.data = torch.nn.Parameter(1000. * torch.ones(1, HIDDEN_STATE_SIZE))
    LSTM.weight_o.data = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(0.))

    # Test the inputs
    new_hidden_state = LSTM(x, hidden_state)[0, :HIDDEN_STATE_SIZE]
    # We expect the input cell state to also be the output cell state
    # So the output hidden state should simply be tanh(input cell state)
    expected_hidden_state = torch.tanh(hidden_state[0, HIDDEN_STATE_SIZE:])

    assert torch.isclose(expected_hidden_state, new_hidden_state).all(), \
        'Output gate test failed with its weights set to 0.'

    # Now turn on the output gate only partially via biases
    partial_bias_value = 0.5
    LSTM.bias_o.data = torch.nn.Parameter(partial_bias_value * torch.ones(1, HIDDEN_STATE_SIZE))

    # Test the inputs
    new_hidden_state = LSTM(x, hidden_state)[0, :HIDDEN_STATE_SIZE]
    # We expect the input cell state to also be the output cell state
    # So the output hidden state should simply be tanh(input cell state)
    expected_hidden_state = torch.tanh(hidden_state[0, HIDDEN_STATE_SIZE:]) * \
                            torch.sigmoid(torch.tensor(partial_bias_value))

    assert torch.isclose(expected_hidden_state, new_hidden_state).all(), \
        'Output gate test failed when gate is partially turned on.'


@pytest.mark.parametrize('x, hidden_state, weight, bias, expected_hidden_state', [
    # Weights are 0
    ([1] * INPUT_SIZE, [7.] * HIDDEN_STATE_SIZE + [0] * HIDDEN_STATE_SIZE, 0., 0.,
     torch.tensor([0.] * 2 * HIDDEN_STATE_SIZE)),
    ([2] * INPUT_SIZE, [8.] * HIDDEN_STATE_SIZE + [0.1] * HIDDEN_STATE_SIZE, 0., 0.5,
     torch.tensor([0.20932218] * HIDDEN_STATE_SIZE + [0.3498951] * HIDDEN_STATE_SIZE)),
    ([3] * INPUT_SIZE, [9.] * HIDDEN_STATE_SIZE + [0.5] * HIDDEN_STATE_SIZE, 0., 1.,
     torch.tensor([0.5315] * HIDDEN_STATE_SIZE + [0.9223] * HIDDEN_STATE_SIZE)),
    ([4] * INPUT_SIZE, [10.] * HIDDEN_STATE_SIZE + [1.5] * HIDDEN_STATE_SIZE, 0., 4.2,
     torch.tensor([0.9710] * HIDDEN_STATE_SIZE + [2.4626] * HIDDEN_STATE_SIZE)),
    ([5] * INPUT_SIZE, [11.] * HIDDEN_STATE_SIZE + [2] * HIDDEN_STATE_SIZE, 0., -0.5,
     torch.tensor([0.1975] * HIDDEN_STATE_SIZE + [0.5806] * HIDDEN_STATE_SIZE)),
    ([6] * INPUT_SIZE, [12.] * HIDDEN_STATE_SIZE + [42] * HIDDEN_STATE_SIZE, 0., -1.5,
     torch.tensor([0.1824] * HIDDEN_STATE_SIZE + [7.4968] * HIDDEN_STATE_SIZE)),
    # Weights and biases are both non-zero
    ([2] * INPUT_SIZE, [8.] * HIDDEN_STATE_SIZE + [0.1] * HIDDEN_STATE_SIZE, 1., 0.5,
     torch.tensor([0.8005] * HIDDEN_STATE_SIZE + [1.1] * HIDDEN_STATE_SIZE)),
    ([3] * INPUT_SIZE, [9.] * HIDDEN_STATE_SIZE + [0.5] * HIDDEN_STATE_SIZE, 4.2, 1,
     torch.tensor([0.9051] * HIDDEN_STATE_SIZE + [1.5] * HIDDEN_STATE_SIZE)),
    ([4] * INPUT_SIZE, [10.] * HIDDEN_STATE_SIZE + [1.5] * HIDDEN_STATE_SIZE, -0.5, 4.2,
     torch.tensor([0.] * HIDDEN_STATE_SIZE + [0.] * HIDDEN_STATE_SIZE)),
    ([5] * INPUT_SIZE, [11.] * HIDDEN_STATE_SIZE + [2] * HIDDEN_STATE_SIZE, -1.5, -0.5,
     torch.tensor([0.] * HIDDEN_STATE_SIZE + [0.] * HIDDEN_STATE_SIZE)),
    ([6] * INPUT_SIZE, [12.] * HIDDEN_STATE_SIZE + [42] * HIDDEN_STATE_SIZE, -10., -1.5,
     torch.tensor([0.] * HIDDEN_STATE_SIZE + [0.] * HIDDEN_STATE_SIZE)),
    # Biases are 0
    ([1] * INPUT_SIZE, [7.] * HIDDEN_STATE_SIZE + [0] * HIDDEN_STATE_SIZE, 0.5, 0.,
     torch.tensor([0.7616] * HIDDEN_STATE_SIZE + [1.] * HIDDEN_STATE_SIZE)),
    ([2] * INPUT_SIZE, [8.] * HIDDEN_STATE_SIZE + [0.1] * HIDDEN_STATE_SIZE, 1., 0.,
     torch.tensor([0.8005] * HIDDEN_STATE_SIZE + [1.1] * HIDDEN_STATE_SIZE)),
    ([3] * INPUT_SIZE, [9.] * HIDDEN_STATE_SIZE + [0.5] * HIDDEN_STATE_SIZE, 4.2, 0.,
     torch.tensor([0.9051] * HIDDEN_STATE_SIZE + [1.5] * HIDDEN_STATE_SIZE)),
    ([4] * INPUT_SIZE, [10.] * HIDDEN_STATE_SIZE + [1.5] * HIDDEN_STATE_SIZE, -0.5, 0.,
     torch.tensor([0.] * HIDDEN_STATE_SIZE + [0.] * HIDDEN_STATE_SIZE)),
    ([5] * INPUT_SIZE, [11.] * HIDDEN_STATE_SIZE + [2] * HIDDEN_STATE_SIZE, -1.5, 0.,
     torch.tensor([0.] * HIDDEN_STATE_SIZE + [0.] * HIDDEN_STATE_SIZE)),
    ([6] * INPUT_SIZE, [12.] * HIDDEN_STATE_SIZE + [42] * HIDDEN_STATE_SIZE, -10., 0.,
     torch.tensor([0.] * HIDDEN_STATE_SIZE + [0.] * HIDDEN_STATE_SIZE)),
])
def test_random_values(x, hidden_state, weight, bias, expected_hidden_state):
    """
    Tests GRU equations for random input, weight and bias values.

    :param x: Input vector
    :param hidden_state: Input hidden state vector
    :param weight: All the weights are set to this value
    :param bias: All the biases are set to this value
    :param expected_hidden_state: Expected value of the concatenated hidden state and memory cell state
    """
    # Convert inputs to tensors with an extra dimension added for batch size
    x = torch.tensor(x).unsqueeze(0).type(torch.float)
    hidden_state = torch.tensor(hidden_state).unsqueeze(0).type(torch.float)
    # Set the weights and biases to the provided value
    for name, w in LSTM.named_parameters():
        if 'weight' in name:
            w.data = torch.nn.Parameter(torch.empty(INPUT_SIZE + HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE).fill_(weight))
        if 'bias' in name:
            w.data = torch.nn.Parameter(bias * torch.ones(1, HIDDEN_STATE_SIZE))
    # Test the inputs
    new_hidden_state = LSTM(x, hidden_state).squeeze()

    assert torch.isclose(expected_hidden_state, new_hidden_state, atol=1e-4).all()


@pytest.mark.parametrize('hidden_state_size, input_size', [(7, 7), (5, 10), (10, 5)])
def test_lstm_initialisation_shapes(hidden_state_size, input_size):
    lstm = LSTMCell(hidden_state_size, input_size)
    for w in (lstm.weight_i, lstm.weight_f, lstm.weight_o, lstm.weight):
        assert list((hidden_state_size + input_size, hidden_state_size)) == list(w.shape)
    for b in (lstm.bias_i, lstm.bias_f, lstm.bias_o, lstm.bias):
        assert [1, hidden_state_size] == list(b.shape)
