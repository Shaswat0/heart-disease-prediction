from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

def quantum_inference(feature_vector):
    num_qubits = min(len(feature_vector), 4)
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(feature_vector[i], i)
    for i in range(num_qubits - 1):
        qc.cz(i, i + 1)
    qc.measure_all()
    simulator = AerSimulator()
    transpiled_qc = transpile(qc, simulator)
    result = simulator.run(transpiled_qc).result()
    counts = result.get_counts()
    probabilities = np.array(list(counts.values())) / sum(counts.values())
    return np.mean(probabilities)
