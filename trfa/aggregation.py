
"""
TRFA Sample-Size-Weighted FedAvg
--------------------------------
Aggregates only the accepted client models.

The trust score is used for screening, not as a second
aggregation weight.

Aggregation:
    w_i = N_i / sum(N_j) for accepted clients only

    W_global = sum(w_i * W_i)
"""

from collections import OrderedDict

import torch


def aggregate_accepted_models(
    client_states,
    sample_sizes,
    accepted_client_ids,
    minimum_accepted_clients=2
):
    """
    Aggregate accepted client model states using sample-size weights.

    Parameters
    ----------
    client_states : dict
        {client_id: model_state_dict}

    sample_sizes : dict
        {client_id: positive training sample count}

    accepted_client_ids : list
        Client IDs that passed TRFA screening.

    minimum_accepted_clients : int
        Minimum number of clients required for aggregation.

    Returns
    -------
    aggregated_state : OrderedDict
        New global model state dictionary.

    aggregation_weights : dict
        Sample-size weights for accepted clients only.
    """

    if not isinstance(client_states, dict) or not client_states:
        raise ValueError("client_states must be a non-empty dictionary.")

    if not isinstance(sample_sizes, dict):
        raise TypeError("sample_sizes must be a dictionary.")

    accepted_ids = list(accepted_client_ids)

    if len(accepted_ids) != len(set(accepted_ids)):
        raise ValueError("Duplicate accepted client IDs were supplied.")

    if len(accepted_ids) < minimum_accepted_clients:
        raise ValueError(
            f"At least {minimum_accepted_clients} accepted clients "
            "are required."
        )

    # Verify that each accepted client has both a model and a sample count.
    for client_id in accepted_ids:
        if client_id not in client_states:
            raise KeyError(
                f"Model state missing for accepted client {client_id}."
            )

        if client_id not in sample_sizes:
            raise KeyError(
                f"Sample count missing for accepted client {client_id}."
            )

        count = sample_sizes[client_id]

        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(
                f"Sample count for client {client_id} must be an integer."
            )

        if count <= 0:
            raise ValueError(
                f"Sample count for client {client_id} must be positive."
            )

    # Rejected clients are deliberately excluded from this list.
    accepted_states = [
        client_states[client_id]
        for client_id in accepted_ids
    ]

    reference_keys = list(accepted_states[0].keys())

    if not reference_keys:
        raise ValueError("The model state dictionary is empty.")

    # Verify all accepted models have identical parameter structure.
    for client_id, state in zip(accepted_ids, accepted_states):
        if list(state.keys()) != reference_keys:
            raise ValueError(
                f"Model parameter keys or ordering differ for client {client_id}."
            )

    for parameter_name in reference_keys:

        reference_tensor = accepted_states[0][parameter_name]

        if not isinstance(reference_tensor, torch.Tensor):
            raise TypeError(
                f"{parameter_name} is not a PyTorch tensor."
            )

        if not torch.is_floating_point(reference_tensor):
            raise TypeError(
                f"{parameter_name} must be a floating-point tensor."
            )

        for client_id, state in zip(accepted_ids, accepted_states):

            tensor = state[parameter_name]

            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"{parameter_name} is not a tensor for client {client_id}."
                )

            if tensor.shape != reference_tensor.shape:
                raise ValueError(
                    f"Parameter shape mismatch for client {client_id}: "
                    f"{parameter_name}"
                )

            if tensor.dtype != reference_tensor.dtype:
                raise ValueError(
                    f"Parameter dtype mismatch for client {client_id}: "
                    f"{parameter_name}"
                )

            if not torch.isfinite(tensor).all().item():
                raise ValueError(
                    f"Non-finite parameter values for client {client_id}: "
                    f"{parameter_name}"
                )

    # Calculate aggregation weights using accepted clients only.
    total_samples = sum(
        sample_sizes[client_id]
        for client_id in accepted_ids
    )

    aggregation_weights = {
        client_id: sample_sizes[client_id] / total_samples
        for client_id in accepted_ids
    }

    if abs(sum(aggregation_weights.values()) - 1.0) > 1e-9:
        raise ArithmeticError("Aggregation weights do not sum to one.")

    # Aggregate every model parameter.
    aggregated_state = OrderedDict()

    for parameter_name in reference_keys:

        reference_tensor = accepted_states[0][parameter_name]

        # Use float64 accumulation for better numerical precision.
        aggregate = torch.zeros(
            reference_tensor.shape,
            dtype=torch.float64,
            device=reference_tensor.device
        )

        for client_id, state in zip(accepted_ids, accepted_states):

            weight = aggregation_weights[client_id]

            aggregate += (
                state[parameter_name].to(torch.float64) * weight
            )

        # Preserve the original model parameter dtype.
        aggregated_state[parameter_name] = aggregate.to(
            dtype=reference_tensor.dtype
        )

    return aggregated_state, aggregation_weights


# ============================================================
# SYNTHETIC AGGREGATION TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("TRFA FEDAVG AGGREGATION TEST")
    print("=" * 65)

    # Three artificial clients.
    client_states = {
        1: {
            "weight": torch.tensor([2.0])
        },
        2: {
            "weight": torch.tensor([4.0])
        },
        3: {
            # This rejected client's value is intentionally very large.
            "weight": torch.tensor([1000.0])
        }
    }

    sample_sizes = {
        1: 100,
        2: 300,
        3: 10000
    }

    # Client 3 is rejected by TRFA.
    accepted_clients = [1, 2]

    global_state, weights = aggregate_accepted_models(
        client_states=client_states,
        sample_sizes=sample_sizes,
        accepted_client_ids=accepted_clients,
        minimum_accepted_clients=2
    )

    print("\nAccepted clients:", accepted_clients)
    print("Rejected client: 3")
    print("Aggregation weights:", weights)

    print(
        "Aggregated parameter:",
        global_state["weight"].item()
    )

    # Expected:
    # (2.0 * 0.25) + (4.0 * 0.75) = 3.5

    expected_value = 3.5

    assert abs(
        global_state["weight"].item() - expected_value
    ) < 1e-6

    assert abs(weights[1] - 0.25) < 1e-9
    assert abs(weights[2] - 0.75) < 1e-9

    assert 3 not in weights

    print("\nRejected client contribution: NONE")
    print("Aggregation weight sum:", sum(weights.values()))

    print("\nTRFA AGGREGATION TEST PASSED")
    print("=" * 65)
