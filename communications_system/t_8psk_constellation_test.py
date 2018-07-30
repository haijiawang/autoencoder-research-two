#LINK TO CODE: https://github.com/veeresht/CommPy/blob/master/commpy/modulation.py

""""
==================================================
Modulation Demodulation (:mod:`commpy.modulation`)
==================================================
.. autosummary::
   :toctree: generated/
   PSKModem             -- Phase Shift Keying (PSK) Modem.
   QAMModem             -- Quadrature Amplitude Modulation (QAM) Modem.
   mimo_ml              -- MIMO Maximum Likelihood (ML) Detection.
"""
from numpy import arange, array, zeros, pi, cos, sin, sqrt, log2, argmin, \
                  hstack, repeat, tile, dot, sum, shape, concatenate, exp, \
                  log, vectorize
from itertools import product
from commpy.utilities import bitarray2dec, dec2bitarray
from numpy.fft import fft, ifft

__all__ = ['PSKModem', 'mimo_ml']


from bitarray import bitarray
arr = bitarray (2**3)
class Modem:

    def modulate(self, input_bits):
        """ Modulate (map) an array of bits to constellation symbols.
        Parameters
        ----------
        input_bits : 1D ndarray of ints
            Inputs bits to be modulated (mapped).
        Returns
        -------
        baseband_symbols : 1D ndarray of complex floats
            Modulated complex symbols.
        """
        mapfunc = vectorize(lambda i:
            self.constellation[bitarray2dec(input_bits[i:i+self.num_bits_symbol])])

        baseband_symbols = mapfunc(arange(0, len(input_bits), self.num_bits_symbol))

        return baseband_symbols

    def demodulate(self, input_symbols, demod_type, noise_var = 0):
        """ Demodulate (map) a set of constellation symbols to corresponding bits.
        Supports hard-decision demodulation only.
        Parameters
        ----------
        input_symbols : 1D ndarray of complex floats
            Input symbols to be demodulated.
        demod_type : string
            'hard' for hard decision output (bits)
            'soft' for soft decision output (LLRs)
        noise_var : float
            AWGN variance. Needs to be specified only if demod_type is 'soft'
        Returns
        -------
        demod_bits : 1D ndarray of ints
            Corresponding demodulated bits.
        """
        if demod_type == 'hard':
            index_list = map(lambda i: argmin(abs(input_symbols[i] - self.constellation)), \
                             range(0, len(input_symbols)))
            demod_bits = hstack(map(lambda i: dec2bitarray(i, self.num_bits_symbol),
                                index_list))
        elif demod_type == 'soft':
            demod_bits = zeros(len(input_symbols) * self.num_bits_symbol)
            for i in arange(len(input_symbols)):
                current_symbol = input_symbols[i]
                for bit_index in arange(self.num_bits_symbol):
                    llr_num = 0
                    llr_den = 0
                    for const_index in self.symbol_mapping:
                        if (const_index >> bit_index) & 1:
                            llr_num = llr_num + exp((-abs(current_symbol - self.constellation[const_index])**2)/noise_var)
                        else:
                            llr_den = llr_den + exp((-abs(current_symbol - self.constellation[const_index])**2)/noise_var)
                    demod_bits[i*self.num_bits_symbol + self.num_bits_symbol - 1 - bit_index] = log(llr_num/llr_den)
        else:
            pass
            # throw an error

        return demod_bits


class PSKModem(Modem):
    """ Creates a Phase Shift Keying (PSK) Modem object. """

    def _constellation_symbol(self, i):
        return cos(2*pi*(i-1)/self.m) + sin(2*pi*(i-1)/self.m)*(0+1j)

    def __init__(self, m):
        """ Creates a Phase Shift Keying (PSK) Modem object.
        Parameters
        ----------
        m : int
            Size of the PSK constellation.
        """
        self.m = m
        self.num_bits_symbol = int(log2(self.m))
        self.symbol_mapping = arange(self.m)
        self.constellation = list(map(self._constellation_symbol,
                                 self.symbol_mapping))


test = Modem()
test_psk = PSKModem(8)
const_sym = test_psk._constellation_symbol(0.2)
print(const_sym)

def ofdm_tx(x, nfft, nsc, cp_length):
    """ OFDM Transmit signal generation """

    nfft = float(nfft)
    nsc = float(nsc)
    cp_length = float(cp_length)
    ofdm_tx_signal = array([])

    for i in range(0, shape(x)[1]):
        symbols = x[:,i]
        ofdm_sym_freq = zeros(nfft, dtype=complex)
        ofdm_sym_freq[1:(nsc/2)+1] = symbols[nsc/2:]
        ofdm_sym_freq[-(nsc/2):] = symbols[0:nsc/2]
        ofdm_sym_time = ifft(ofdm_sym_freq)
        cp = ofdm_sym_time[-cp_length:]
        ofdm_tx_signal = concatenate((ofdm_tx_signal, cp, ofdm_sym_time))

    return ofdm_tx_signal

def ofdm_rx(y, nfft, nsc, cp_length):
    """ OFDM Receive Signal Processing """

    num_ofdm_symbols = int(len(y)/(nfft + cp_length))
    x_hat = zeros([nsc, num_ofdm_symbols], dtype=complex)

    for i in range(0, num_ofdm_symbols):
        ofdm_symbol = y[i*nfft + (i+1)*cp_length:(i+1)*(nfft + cp_length)]
        symbols_freq = fft(ofdm_symbol)
        x_hat[:,i] = concatenate((symbols_freq[-nsc/2:], symbols_freq[1:(nsc/2)+1]))

    return x_hat

def mimo_ml(y, h, constellation):
    """ MIMO ML Detection.
    parameters
    ----------
    y : 1D ndarray of complex floats
        Received complex symbols (shape: num_receive_antennas x 1)
    h : 2D ndarray of complex floats
        Channel Matrix (shape: num_receive_antennas x num_transmit_antennas)
    constellation : 1D ndarray of complex floats
        Constellation used to modulate the symbols
    """
    m = len(constellation)
    x_ideal = array([tile(constellation, m), repeat(constellation, m)])
    y_vector = tile(y, m*m)
    min_idx = argmin(sum(abs(y_vector - dot(h, x_ideal)), axis=0))
    x_r = x_ideal[:, min_idx]

    return x_r