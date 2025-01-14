import numpy as np
import scipy as sc
import pandas as pd
import qutip as qt
from time import time
from random import sample
import seaborn as sns
import matplotlib.pyplot as plt

qubit0 = qt.basis(2, 0)
qubit1 = qt.basis(2, 1)
qubit0mat = qubit0 * qubit0.dag()
qubit1mat = qubit1 * qubit1.dag()

def partialTraceKeep(obj, keep):
    return obj.ptrace(keep) if len(keep) != len(obj.dims[0]) else obj

def partialTraceRem(obj, rem):
    keep = list(range(len(obj.dims[0])))
    for x in sorted(rem, reverse=True):
        keep.pop(x)
    return obj.ptrace(keep) if len(keep) != len(obj.dims[0]) else obj

def swappedOp(obj, i, j):
    if i == j:
        return obj
    permute = list(range(len(obj.dims[0])))
    permute[i], permute[j] = permute[j], permute[i]
    return obj.permute(permute)

def tensoredId(N):
    res = qt.qeye(2**N)
    dims = [[2] * N, [2] * N]
    res.dims = dims
    return res

def tensoredQubit0(N):
    res = qt.fock(2**N).proj()
    dims = [[2] * N, [2] * N]
    res.dims = dims
    return res

def randomQubitUnitary(numQubits):
    dim = 2**numQubits
    res = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    res = sc.linalg.orth(res)
    res = qt.Qobj(res)
    dims = [[2] * numQubits, [2] * numQubits]
    res.dims = dims
    return res

def randomQubitState(numQubits):
    dim = 2**numQubits
    res = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    res = (1 / np.linalg.norm(res)) * res
    res = qt.Qobj(res)
    dims = [[2] * numQubits, [1] * numQubits]
    res.dims = dims
    return res

def noisyState(inputState, noiseLevel=0.1):
    noise = (np.random.normal(size=inputState.shape) + 1j * np.random.normal(size=inputState.shape)) * noiseLevel
    noisy_state = inputState.full() + noise
    noisy_state = noisy_state / np.linalg.norm(noisy_state)
    return qt.Qobj(noisy_state, dims=inputState.dims)

def randomTrainingData(unitary, N):
    numQubits = len(unitary.dims[0])
    trainingData = []
    for _ in range(N):
        t = randomQubitState(numQubits)
        ut = unitary * t
        if not isinstance(t, qt.Qobj) or not isinstance(ut, qt.Qobj):
            raise ValueError("Training data contains non-Qobj elements.")
        trainingData.append([t, ut])
    return trainingData

def randomNetwork(qnnArch, numTrainingPairs):
    assert qnnArch[0] == qnnArch[-1], "Not a valid QNN-Architecture."
    networkUnitary = randomQubitUnitary(qnnArch[-1])
    networkTrainingData = randomTrainingData(networkUnitary, numTrainingPairs)
    networkUnitaries = [[]]
    for l in range(1, len(qnnArch)):
        numInputQubits = qnnArch[l-1]
        numOutputQubits = qnnArch[l]
        networkUnitaries.append([])
        for j in range(numOutputQubits):
            unitary = randomQubitUnitary(numInputQubits+1)
            if numOutputQubits-1 != 0:
                unitary = qt.tensor(randomQubitUnitary(numInputQubits+1), tensoredId(numOutputQubits-1))
            unitary = swappedOp(unitary, numInputQubits, numInputQubits + j)
            networkUnitaries[l].append(unitary)
    return (qnnArch, networkUnitaries, networkTrainingData, networkUnitary)

def costFunction(trainingData, outputStates):
    costSum = 0
    for i in range(len(trainingData)):
        t = trainingData[i][1]
        o = outputStates[i]
        if not isinstance(t, qt.Qobj) or not isinstance(o, qt.Qobj):
            raise ValueError(f"TrainingData[{i}] or OutputState[{i}] is not a Qobj.")
        term = (t.dag() * o * t)
        if isinstance(term, qt.Qobj):
            costSum += term.tr().real
        else:
            costSum += term.real
    return costSum / len(trainingData)

def makeLayerChannel(qnnArch, unitaries, l, inputState):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]
    if inputState.type == 'ket':
        inputState = inputState * inputState.dag()
    state = qt.tensor(inputState, tensoredQubit0(numOutputQubits))
    layerUni = unitaries[l][0].copy()
    for i in range(1, numOutputQubits):
        layerUni = unitaries[l][i] * layerUni
    result = layerUni * state * layerUni.dag()
    return result.ptrace(list(range(numInputQubits, numInputQubits + numOutputQubits)))

def makeAdjointLayerChannel(qnnArch, unitaries, l, outputState):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]
    inputId = tensoredId(numInputQubits)
    state1 = qt.tensor(inputId, tensoredQubit0(numOutputQubits))
    state2 = qt.tensor(inputId, outputState)
    layerUni = unitaries[l][0].copy()
    for i in range(1, numOutputQubits):
        layerUni = unitaries[l][i] * layerUni
    return partialTraceKeep(state1 * layerUni.dag() * state2 * layerUni, list(range(numInputQubits)))

def feedforward(qnnArch, unitaries, trainingData):
    storedStates = []
    for x in range(len(trainingData)):
        currentState = trainingData[x][0] * trainingData[x][0].dag()
        layerwiseList = [currentState]
        for l in range(1, len(qnnArch)):
            currentState = makeLayerChannel(qnnArch, unitaries, l, currentState)
            layerwiseList.append(currentState)
        storedStates.append(layerwiseList)
    return storedStates

def makeUpdateMatrix(qnnArch, unitaries, trainingData, storedStates, lda, ep, l, j):
    numInputQubits = qnnArch[l-1]
    summ = 0
    for x in range(len(trainingData)):
        firstPart = updateMatrixFirstPart(qnnArch, unitaries, storedStates, l, j, x)
        secondPart = updateMatrixSecondPart(qnnArch, unitaries, trainingData, l, j, x)
        mat = qt.commutator(firstPart, secondPart)
        keep = list(range(numInputQubits))
        keep.append(numInputQubits + j)
        mat = partialTraceKeep(mat, keep)
        summ = summ + mat
    summ = (-ep * (2**numInputQubits)/(lda*len(trainingData))) * summ
    return summ.expm()

def updateMatrixFirstPart(qnnArch, unitaries, storedStates, l, j, x):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]
    state = qt.tensor(storedStates[x][l-1], tensoredQubit0(numOutputQubits))
    productUni = unitaries[l][0]
    for i in range(1, j+1):
        productUni = unitaries[l][i] * productUni
    return productUni * state * productUni.dag()

def updateMatrixSecondPart(qnnArch, unitaries, trainingData, l, j, x):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]
    state = trainingData[x][1] * trainingData[x][1].dag()
    for i in range(len(qnnArch)-1, l, -1):
        state = makeAdjointLayerChannel(qnnArch, unitaries, i, state)
    state = qt.tensor(tensoredId(numInputQubits), state)
    productUni = tensoredId(numInputQubits + numOutputQubits)
    for i in range(j+1, numOutputQubits):
        productUni = unitaries[l][i] * productUni
    return productUni.dag() * state * productUni

def makeUpdateMatrixTensored(qnnArch, unitaries, trainingData, storedStates, lda, ep, l, j):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]
    res = makeUpdateMatrix(qnnArch, unitaries, trainingData, storedStates, lda, ep, l, j)
    if numOutputQubits-1 != 0:
        res = qt.tensor(res, tensoredId(numOutputQubits-1))
    return swappedOp(res, numInputQubits, numInputQubits + j)

def qnnTraining(qnnArch, initialUnitaries, trainingData, lda, ep, trainingRounds, alert=0):
    s = 0
    currentUnitaries = initialUnitaries
    storedStates = feedforward(qnnArch, currentUnitaries, trainingData)
    outputStates = [storedStates[k][-1] for k in range(len(storedStates))]
    plotlist = [[s], [costFunction(trainingData, outputStates)]]
    for k in range(trainingRounds):
        if alert > 0 and k % alert == 0:
            print(f"In training round {k}")
        newUnitaries = [layer.copy() for layer in currentUnitaries]
        for l in range(1, len(qnnArch)):
            numOutputQubits = qnnArch[l]
            for j in range(numOutputQubits):
                newUnitaries[l][j] = makeUpdateMatrixTensored(qnnArch, currentUnitaries, trainingData, storedStates, lda, ep, l, j) * currentUnitaries[l][j]
        s += ep
        currentUnitaries = newUnitaries
        storedStates = feedforward(qnnArch, currentUnitaries, trainingData)
        outputStates = [storedStates[m][-1] for m in range(len(storedStates))]
        plotlist[0].append(s)
        plotlist[1].append(costFunction(trainingData, outputStates))
    print(f"Trained {trainingRounds} rounds for a {qnnArch} network and {len(trainingData)} training pairs")
    return [plotlist, currentUnitaries]

def detectAnomaly(qnnArch, trainedUnitaries, inputState, threshold):
    currentState = inputState
    for l in range(1, len(qnnArch)):
        currentState = makeLayerChannel(qnnArch, trainedUnitaries, l, currentState)
    
    
    currentState = currentState.proj() if currentState.isket else currentState
    inputState = inputState.proj() if inputState.isket else inputState

    try:
        fidelity = abs((currentState * inputState).tr().real)
        trace_distance = 1 - abs(currentState.tr().real)
        entropy = -abs(currentState.tr().real) * np.log(abs(currentState.tr().real) + 1e-10)
        anomalyScore = trace_distance + (1 - fidelity) + entropy
    except Exception as e:
        print(f"Error calculating anomaly score: {e}")
        anomalyScore = 0.0  

    return max(anomalyScore, 1e-6)


def adjustThreshold(anomaly_scores, percentile=80):
    return np.percentile(anomaly_scores, percentile)

def adjustPolicy(currentPolicy, anomalyScore, threshold):
    if anomalyScore > threshold:
        return "Restricted"
    elif anomalyScore > threshold/2:
        return "Monitored"
    elif anomalyScore > threshold/4:
        return "Inspected"
    return "Open"

def simulateTraffic(qnnArch, trainedUnitaries, numSamples, anomalyThreshold, policyThreshold, noiseLevel=0.2):
    for _ in range(numSamples):
        inputState = randomQubitState(qnnArch[0])
        if noiseLevel > 0.0:
            inputState = noisyState(inputState, noiseLevel)
        anomalyScore = detectAnomaly(qnnArch, trainedUnitaries, inputState, anomalyThreshold)
        policy = adjustPolicy("Default", anomalyScore, policyThreshold)
        print(f"Anomaly Score: {anomalyScore:.4f}, Policy: {policy}")

def load_cesnet_data(file_path, num_samples=5):
    df = pd.read_csv(file_path)
    df = df.head(num_samples)
    features = [
        'n_flows', 'n_packets', 'n_bytes',
        'n_dest_ip', 'n_dest_ports',
        'tcp_udp_ratio_packets',
        'dir_ratio_packets',
        'avg_duration'
    ]
    data = df[features].values
    std = np.std(data, axis=0)
    std = np.where(std < 1e-10, 1e-10, std)
    normalized_data = (data - np.mean(data, axis=0)) / std
    return normalized_data, df

def preprocess_features(df):
    df['bytes_per_packet'] = df['n_bytes'] / df['n_packets']
    df['packets_per_flow'] = df['n_packets'] / df['n_flows']
    df['ports_per_ip'] = df['n_dest_ports'] / df['n_dest_ip']
    return df

def encode_traffic_to_quantum(normalized_data):
    """Convert normalized traffic data to quantum states"""
    quantum_states = []
    for sample in normalized_data:
        norm = np.linalg.norm(sample)
        if norm != 0:
            quantum_state = sample / norm
        else:
            quantum_state = sample
        # Ensure the data fits into quantum state dimensions
        num_qubits = int(np.ceil(np.log2(len(sample))))
        padded_state = np.zeros(2**num_qubits)
        padded_state[:len(sample)] = quantum_state
        padded_state = padded_state / np.linalg.norm(padded_state)
        quantum_states.append(qt.Qobj(padded_state.reshape(-1, 1), 
                            dims=[[2]*num_qubits, [1]*num_qubits]))
    return quantum_states


def prepare_training_data(quantum_states, noise_level=0.1):
    """Prepare training data with noisy target states for anomaly detection."""
    training_data = []
    for state in quantum_states:
        noisy_target = noisyState(state, noise_level)  # Add noise to the target
        training_data.append([state, noisy_target])
    return training_data


print("Loading CESNET dataset...")
data_path = "ip_addresses_sample/agg_10_minutes/11.csv" 
normalized_data, raw_df = load_cesnet_data(data_path)
quantum_states = encode_traffic_to_quantum(normalized_data)
training_data = prepare_training_data(quantum_states)


num_features = normalized_data.shape[1]
num_qubits = int(np.ceil(np.log2(num_features)))


networks = [
    ([num_qubits, num_qubits*2, num_qubits], len(training_data)//2),
    ([num_qubits, num_qubits+1, num_qubits], len(training_data)//2),
    ([num_qubits, num_qubits], len(training_data)//2)
]
simulation_results = []

for arch, num_pairs in networks:
    print(f"\nTraining network {arch} with {num_pairs} training pairs...")
    network = randomNetwork(arch, num_pairs)
    trainedUnitaries = qnnTraining(network[0], network[1], training_data[:num_pairs], 1, 0.1, 100)[1]
    
    print(f"\nTesting network {arch} with real traffic data...")
    # Test with remaining data
    test_data = training_data[num_pairs:]
    anomaly_scores = []
    for test_state, _ in test_data:
        currentState = test_state
        for l in range(1, len(arch)):
            currentState = makeLayerChannel(arch, trainedUnitaries, l, currentState)
        try:
            anomaly_score = 1 - abs(currentState.tr().real)  # Add abs()
            if np.isnan(anomaly_score):
                anomaly_score = 0.0
        except:
            anomaly_score = 0.0
        anomaly_scores.append(anomaly_score)
    
  
    threshold = np.percentile(anomaly_scores, 95)  
    
    print(f"Testing traffic patterns...")
    for i, score in enumerate(anomaly_scores[:20]):  
        policy = "Restricted" if score > threshold else "Open"
        print(f"Sample {i+1} - Anomaly Score: {score:.4f}, Policy: {policy}")



if any(not np.isnan(score) for score in anomaly_scores):
    plt.figure(figsize=(10, 6))
    valid_scores = [score for score in anomaly_scores if not np.isnan(score)]
    plt.hist(anomaly_scores, bins=20, alpha=0.7)
    plt.title(f"Anomaly Score Distribution for Network {arch}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.show()
else:
    print("No valid anomaly scores to plot")

data_path = "agg_10_minutes/11.csv"  
network121 = randomNetwork([1, 2, 1], 10)
trainedUnitaries = qnnTraining(network121[0], network121[1], network121[2], 1, 0.1, 500)[1]
simulateTraffic(network121[0], trainedUnitaries, 20, 0.5, 0.7)



network121 = randomNetwork([1, 2, 1], 10)
trainedUnitaries = qnnTraining(network121[0], network121[1], network121[2], 1, 0.1, 500)[1]
simulateTraffic(network121[0], trainedUnitaries, 20, 0.5, 0.7)

networks = [
    ([3, 6, 3], 10), 
    ([3, 4, 3], 10),
    ([3, 3], 10)
]

for arch, num_pairs in networks:
    print(f"\nSimulation for network {arch} with {num_pairs} training pairs:")
    network = randomNetwork(arch, num_pairs)
    trainedUnitaries = qnnTraining(network[0], network[1], network[2], 1, 0.1, 200)[1]

   
    for sample_id in range(1, 21): 
        inputState = randomQubitState(arch[0])
        inputState = noisyState(inputState, noiseLevel=0.2)
        anomalyScore = detectAnomaly(arch, trainedUnitaries, inputState, threshold=0.5)
        policy = adjustPolicy("Default", anomalyScore, 0.7)
        
      
        print(f"Sample {sample_id} - Anomaly Score: {anomalyScore:.4f}, Policy: {policy}")
        simulation_results.append([sample_id, str(arch), anomalyScore, policy])


simulation_df = pd.DataFrame(simulation_results, columns=["Sample", "Network Architecture", "Anomaly Score", "Policy"])


simulation_df.to_csv("simulation_results.csv", index=False)
print("Simulation results saved to simulation_results.csv")


unique_networks = simulation_df["Network Architecture"].unique()

for network in unique_networks:
    network_data = simulation_df[simulation_df["Network Architecture"] == network]
    
    
    plt.figure(figsize=(10, 6))
    plt.hist(network_data["Anomaly Score"], bins=10, alpha=0.7)
    plt.title(f"Anomaly Score Distribution for Network {network}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

   =
    plt.figure(figsize=(8, 6))
    policy_counts = network_data["Policy"].value_counts()
    plt.bar(policy_counts.index, policy_counts.values, alpha=0.7)
    plt.title(f"Policy Distribution for Network {network}")
    plt.xlabel("Policy")
    plt.ylabel("Count") 
    plt.grid(True)
    plt.show()

   
    pivot_table = network_data.pivot(index="Sample", columns="Policy", values="Anomaly Score")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Anomaly Score'})
    plt.title(f"Heatmap of Anomaly Scores for Network {network}")
    plt.xlabel("Policy")
    plt.ylabel("Sample")
    plt.show()

    
    plt.figure(figsize=(10, 6))
    anomaly_scores_sorted = network_data["Anomaly Score"].sort_values()
    plt.plot(anomaly_scores_sorted, np.arange(1, len(anomaly_scores_sorted) + 1) / len(anomaly_scores_sorted), marker='o')
    plt.title(f"Cumulative Distribution of Anomaly Scores for Network {network}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Cumulative Probability")
    plt.grid(True)
    plt.show()
