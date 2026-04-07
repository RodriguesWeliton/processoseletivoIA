# Relatório do Candidato

O arquivo `README.md` é o relatório final do desafio.

Preencha todas as seções de forma clara e objetiva.

> 💡 Dica: não é necessário um relatório extenso. O mais importante é demonstrar clareza nas decisões técnicas.

## 👤 Identificação

Nome Completo: Seu Nome Completo

## 1️⃣ Resumo da Arquitetura do Modelo

A arquitetura implementada em `train_model.py` é uma CNN leve projetada para MNIST.
Ela utiliza duas camadas convolucionais `Conv2D` com ativação ReLU, cada uma seguida de `MaxPooling2D`.
Após a segunda camada de pooling, há um `Flatten`, uma camada `Dense` com 128 neurônios e ReLU,
`Dropout(0.5)` para regularização e uma camada de saída `Dense(10, activation="softmax")`
para classificação dos dígitos de 0 a 9.

## 2️⃣ Bibliotecas Utilizadas

- `tensorflow==2.17.0`
- `numpy==1.26.4`

## 3️⃣ Técnica de Otimização do Modelo

No `optimize_model.py`, o modelo salvo em `mnist_cnn.h5` é carregado com `tf.keras.models.load_model`
e convertido para TensorFlow Lite usando `TFLiteConverter.from_keras_model(model)`.
A otimização utilizada foi `converter.optimizations = [tf.lite.Optimize.DEFAULT]`,
que aplica quantização de alcance dinâmico (dynamic range quantization) para reduzir
o tamanho do modelo sem precisar de calibração adicional.

## 4️⃣ Resultados Obtidos

- Acurácia de teste do modelo: **99.02%**
- Modelo Keras salvo em `mnist_cnn.h5`
- Modelo otimizando salvo em `mnist_cnn.tflite`
- Tamanho aproximado do `.tflite`: **232 KB**

## 5️⃣ Comentários Adicionais (Opcional)

- Dificuldades encontradas: equilibrar um modelo leve com desempenho consistente no MNIST.
- Decisões técnicas importantes: usei uma arquitetura simples de CNN e quantização dinâmica para priorizar eficiência.
- Limitações do modelo: não foi implementada quantização inteira completa nem aumento de dados avançado.
- Aprendizados durante o desafio: o fluxo completo de treinamento, salvamento e conversão é essencial para
  entregar um modelo pronto para uso em Edge AI.
