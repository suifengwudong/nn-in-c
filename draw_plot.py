# plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'SimHei'

TRAINING_LOG = 'build/mnist_loss.csv'
df = pd.read_csv(TRAINING_LOG)

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.show()