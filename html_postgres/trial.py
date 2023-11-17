import math
import multiprocessing
import itertools
from timeit import Timer


def check_prime(n):
    if n % 2 == 0:
        return False
    return all(n % i != 0 for i in xrange(3, int(math.sqrt(n)) + 1, 2))


def primes_sequential():
    primes = []
    number_range = xrange(100000000, 101000000)
    for possible_prime in number_range:
        if check_prime(possible_prime):
            primes.append(possible_prime)
    print len(primes), primes[:10], primes[-10:]


def primes_parallel():
    number_range = xrange(100000000, 101000000)
    pool = multiprocessing.Pool(4)  # for each core
    output = pool.map(check_prime, number_range)  # a list of true false
    primes = [p for p in itertools.compress(number_range, output)]
    print len(primes), primes[:10], primes[-10:]


if __name__ == "__main__":
    t = Timer(lambda: primes_sequential())
    print "Completed sequential in %s seconds." % t.timeit(1)

    t = Timer(lambda: primes_parallel())
    print "Completed parallel in %s seconds." % t.timeit(1)
