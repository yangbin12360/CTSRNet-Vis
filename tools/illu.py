from tslearn.piecewise import PiecewiseAggregateApproximation, SymbolicAggregateApproximation
from pyts.approximation import DiscreteFourierTransform
from pyts.approximation import SymbolicAggregateApproximation as SAX_Alpha
from scipy.stats import norm


import matplotlib.pyplot as plt
import numpy as np

n_samples, n_timestamps, n_embedding = 100, 48, 16

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_timestamps)

# DFT transformation
n_coefs = n_embedding
dft = DiscreteFourierTransform(n_coefs=n_coefs, norm_mean=False,
                               norm_std=False)
X_dft = dft.fit_transform(X)

# Compute the inverse transformation
if n_coefs % 2 == 0:
    real_idx = np.arange(1, n_coefs, 2)
    imag_idx = np.arange(2, n_coefs, 2)
    X_dft_new = np.c_[
        X_dft[:, :1],
        X_dft[:, real_idx] + 1j * np.c_[X_dft[:, imag_idx],
                                        np.zeros((n_samples, ))]
    ]
else:
    real_idx = np.arange(1, n_coefs, 2)
    imag_idx = np.arange(2, n_coefs + 1, 2)
    X_dft_new = np.c_[
        X_dft[:, :1],
        X_dft[:, real_idx] + 1j * X_dft[:, imag_idx]
    ]
X_irfft = np.fft.irfft(X_dft_new, n_timestamps)

# PAA transformation
paa = PiecewiseAggregateApproximation(n_segments=n_embedding)
X_paa = paa.inverse_transform(paa.fit_transform(X))

# SAX transformation
n_sax_symbols = 8
sax = SymbolicAggregateApproximation(n_segments=n_embedding,
                                     alphabet_size_avg=n_sax_symbols)
X_s = sax.fit_transform(X)
X_sax = sax.inverse_transform(X_s)

sax_alpha = SAX_Alpha(n_bins=n_sax_symbols, strategy="normal")
X_sax_alpha = sax_alpha.fit_transform(X_paa[:, :, 0])
bins = norm.ppf(np.linspace(0, 1, n_sax_symbols+1)[0: -1])
bottom_bool = np.r_[True, X_sax_alpha[0, 1:] > X_sax_alpha[0, : -1]]

# Show the results for the first time series
plt.figure()
plt.subplot(2, 2, 1)
plt.plot(X[0], 'o--', ls="solid", ms=0, label='Original')
plt.xlabel('Time', fontsize=8)
plt.title('Original', fontsize=10)

plt.subplot(2, 2, 2)
plt.plot(X[0], 'o--', label='Original', ms=0, ls="solid", alpha=0.4)
plt.plot(X_irfft[0], 'o--', ls="solid", ms=0,
         label='DFT - {0} coefs'.format(n_coefs))
plt.legend(loc='best', fontsize=8)
plt.xlabel('Time', fontsize=8)
plt.title('Discrete Fourier Transform', fontsize=10)

plt.subplot(2, 2, 3)
plt.plot(X[0], 'o--', label='Original', ms=0, ls="solid", alpha=0.4)
plt.plot(X_paa[0], 'o--', ls="solid", ms=0, label='PAA')
plt.xlabel('Time', fontsize=8)
plt.legend(loc='best', fontsize=8)
plt.title('Piecewise Aggregate Approximation', fontsize=10)

plt.subplot(2, 2, 4)
plt.plot(X[0], 'o--', label='Original', ms=0, ls="solid", alpha=0.4)
plt.plot(X_sax[0], 'o--', ls="solid", ms=0,
         label='SAX - {0} bins'.format(n_sax_symbols))
for x, y, s, bottom in zip(range(n_timestamps), X_paa[0], X_sax_alpha[0], bottom_bool):
    if x % int(n_timestamps / n_embedding) == 0:
        va = 'bottom' if bottom else 'top'
        plt.text(x, y, s, ha='center', va=va, fontsize=14, color='#ff7f0e')
plt.legend(loc='best', fontsize=8)
plt.xlabel('Time', fontsize=8)
plt.title('Symbolic Aggregate Approximation', fontsize=10)

plt.tight_layout()
plt.show()
