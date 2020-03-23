import numpy, os
with open(os.path.join(os.path.dirname(__file__), "random.npy"), "rb") as f: RANDOM = numpy.load(f)

def seeded_randint(seed, min, max):
    return numpy.cast[numpy.int64](RANDOM[seed % RANDOM.shape[0]] % (max - min) + min)