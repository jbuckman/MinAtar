import numpy
with open("minatar/random.npy", "rb") as f: RANDOM = numpy.load(f)

def seeded_randint(seed, min, max):
    return numpy.cast[numpy.int64](RANDOM[seed % RANDOM.shape[0]] % (max - min) + min)