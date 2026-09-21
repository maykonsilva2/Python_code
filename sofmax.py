# pip install numpy
import numpy as np

def softmax(z):
    """
    Calcula o softmax de um vetor z
    """
    # Subtrair o máximo para estabilidade numérica
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)


# Teste 2: Classificação de dígitos (0-9)
z2 = np.array([0.5, 1.2, -0.3, 2.1, 0.8, -1.0, 1.5, 0.2, 0.9, 0.1])
probs2 = softmax(z2)
print("Exemplo 2 (Dígitos):")
for i in range(10):
    print(f"Dígito {i}: {probs2[i]:.3f} ({probs2[i]*100:.1f}%)")

# Previsão: 30
# Confiança: 33.7%
print(f"\nPrevisão: {np.argmax(probs2)}")
print(f"Confiança: {probs2[np.argmax(probs2)]*100:.1f}%")
