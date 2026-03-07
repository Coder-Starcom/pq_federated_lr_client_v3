## PQ-Federated Logistic Regression — Core Architecture

---

# 1. Overview

The `/core` module implements a complete federated learning framework for binary logistic regression.

This core is designed with the following principles:

- Clear separation of concerns
- Strict abstraction boundaries
- Extensibility for cryptographic integration
- Research-grade modularity
- Deterministic and testable components

The system follows the Federated Averaging (FedAvg) algorithm and simulates a distributed training environment in which:

- A global model is maintained centrally.
- Multiple clients train locally on private data.
- Only model parameters are shared.
- Aggregation occurs at the server.

This architecture is intentionally designed to later integrate post-quantum secure encryption of parameters before transmission.

---

# 2. High-Level Architecture

The federated system consists of the following logical layers:

```
                 ┌─────────────────────────┐
                 │     FederatedServer     │
                 │  (Global Model Holder)  │
                 └────────────┬────────────┘
                              │ Broadcast
                              ▼
         ┌─────────────────────────────────────────┐
         │              Clients                    │
         │  ┌──────────┐  ┌──────────┐  ┌────────┐ │
         │  │ Client 1 │  │ Client 2 │  │Client N│ │
         │  └──────────┘  └──────────┘  └────────┘ │
         └─────────────────────────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │       Aggregator        │
                 │     (FedAvg Logic)      │
                 └─────────────────────────┘
```

Core components:

- Models
- Optimization
- Federated coordination
- Metrics

---

# 3. Core Components

---

## 3.1 BaseModel

Location:

```
core/models/base_model.py
```

Purpose:
Defines the abstract interface for any trainable model.

Responsibilities:

- Store weights and bias
- Define forward pass interface
- Define loss computation interface
- Define gradient computation interface
- Provide parameter management utilities

Abstract Methods:

- `forward(X)`
- `compute_loss(y_true, y_pred)`
- `compute_gradients(X, y_true)`

Concrete Utilities:

- `get_parameters()`
- `set_parameters()`
- `update_parameters()`

Design Rationale:

All future models (e.g., neural networks, linear models, encrypted models) can inherit from this base without modifying the federated layer.

---

## 3.2 LogisticRegression

Location:

```
core/models/logistic_regression.py
```

Implements binary logistic regression from scratch.

Features:

- Sigmoid activation
- Binary cross-entropy loss
- Analytical gradient computation
- Numerically stable implementation

Parameter Structure:

- weights → 1D numpy array (n_features,)
- bias → scalar

Gradient Output:

- weight_gradients → 1D numpy array
- bias_gradient → float

Design Note:

Weights are kept 1D to maintain compatibility with:

- Aggregator
- Optimizer
- Server
- Encryption layer (future)

---

## 3.3 Optimizer Interface

Location:

```
core/optimization/optimizer.py
```

Defines the abstract optimizer contract.

Required method:

```
step(weights, bias, weight_gradients, bias_gradient)
```

Returns updated parameters.

This allows plug-and-play replacement of:

- Gradient Descent
- Momentum
- Adam
- Encrypted Optimizers (future work)

---

## 3.4 GradientDescent

Location:

```
core/optimization/gradient_descent.py
```

Implements batch gradient descent.

Update rule:

```
w = w - lr * dW
b = b - lr * db
```

The optimizer does not store model state.
It is stateless and purely functional.

---

## 3.5 FederatedClient

Location:

```
core/federated/client.py
```

Represents one federated participant.

Responsibilities:

- Maintain local dataset
- Hold local model instance
- Perform local training
- Return updated parameters to server

Core Method:

```
train_local(local_epochs)
```

Training Procedure:

1. Receive global parameters
2. Perform local gradient descent for given epochs
3. Return updated parameters

Important Property:

The client never shares raw data.
Only model parameters are returned.

This property enables secure federated learning.

---

## 3.6 Aggregator

Location:

```
core/federated/aggregator.py
```

Implements Federated Averaging (FedAvg).

Aggregation Rule:

For N clients:

```
W_global = (1/N) * Σ W_client_i
b_global = (1/N) * Σ b_client_i
```

This is a simple arithmetic mean of parameters.

Future Extension:

This module is the exact insertion point for:

- Secure aggregation
- Homomorphic addition
- Post-quantum encrypted averaging

---

## 3.7 FederatedServer

Location:

```
core/federated/server.py
```

Central coordinator of federated learning.

Responsibilities:

- Maintain global model
- Broadcast parameters
- Trigger local client training
- Aggregate client updates
- Update global model

Core Method:

```
train_one_round(local_epochs)
```

Round Procedure:

1. Broadcast global parameters to clients
2. Each client trains locally
3. Collect updated parameters
4. Aggregate parameters
5. Update global model

The server does NOT:

- Access raw client data
- Perform gradient computation
- Apply optimizer logic

It only coordinates and aggregates.

---

## 3.8 FederatedRound

Location:

```
core/federated/round.py
```

Encapsulates a single federated training round.

Provides:

- Execution wrapper
- Optional evaluation
- Metrics dictionary

Purpose:

Improves modularity and logging control.
Useful for experiment orchestration.

---

## 3.9 Metrics Module

Location:

```
core/metrics.py
```

Provides binary classification metrics:

- Accuracy
- Binary Cross Entropy
- Precision
- Recall
- F1-score

All metrics:

- Flatten inputs
- Validate shape
- Return float values

This module is independent of model implementation.

---

# 4. Federated Training Lifecycle

Complete training flow:

### Step 1 — Initialization

- Create global model
- Create clients with local data
- Initialize server

### Step 2 — Training Loop

For each round:

1. Server broadcasts parameters
2. Clients perform local training
3. Clients return updated parameters
4. Server aggregates parameters
5. Global model updated

### Step 3 — Evaluation

Server evaluates global model on test data.

---

# 5. Design Principles

The `/core` module follows:

### 1. Abstraction-Driven Design

All components interact through interfaces.

### 2. Stateless Aggregation

Aggregator has no internal state.

### 3. Deterministic Parameter Flow

Only parameters move between server and clients.

### 4. Data Privacy Separation

Clients never expose raw datasets.

### 5. Extensibility

Encryption layer can be added without modifying model logic.

---

# 6. Encryption Integration Point (Future Work)

Planned architecture:

```
Client
  ↓
Encrypt(weights, bias)
  ↓
Server receives encrypted parameters
  ↓
Secure Aggregation
  ↓
Decrypt aggregated result
  ↓
Update global model
```

Minimal modification required:

- Wrap `client.train_local()` output
- Replace `Aggregator.average_parameters()` with secure equivalent

Core ML logic remains unchanged.

---

# 7. Why This Architecture is Research-Grade

1. Clean separation between:
   - Model
   - Optimization
   - Federated coordination
   - Metrics

2. Clear abstraction layers

3. Deterministic and testable components

4. Easy integration with:
   - Differential privacy
   - Secure multiparty computation
   - Post-quantum cryptography

---

# 8. Current Capability

The core system:

- Trains logistic regression in federated setup
- Uses multiple simulated clients
- Aggregates parameters via FedAvg
- Achieves convergence comparable to centralized training
- Fully modular and extensible

---

# 9. Summary

The `/core` module implements a clean, extensible, and academically structured federated learning framework built from scratch.

It successfully:

- Separates local and global learning
- Preserves client data privacy
- Implements FedAvg correctly
- Maintains extensibility for post-quantum secure aggregation

This forms the foundational layer for PQ-Federated Learning.
