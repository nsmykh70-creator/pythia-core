import unittest, random
from ps9_5_discovery_engine import *
class Tests(unittest.TestCase):
 def test_roundtrips(self):
  rng=random.Random(1234)
  for n in [0,1,2,7,8,9,33,1000]:
   d=bytes(rng.randrange(256) for _ in range(n))
   for lag in [1,2,4,8]:
    self.assertEqual(delta_inverse(delta(d,lag),lag),d)
    self.assertEqual(xor_lag_inverse(xor_lag(d,lag),lag),d)
   for s in [2,4,8,16]: self.assertEqual(transpose_inverse(transpose_lanes(d,s),s),d)
   self.assertEqual(bitplane_inverse(bitplane(d),n),d)
 def test_discovery_deterministic(self):
  d=bytes((i*7+i//3)%256 for i in range(4096))
  a=discover(d); b=discover(d)
  self.assertEqual([(x.name,x.params,x.score_bits) for x in a],[(x.name,x.params,x.score_bits) for x in b])
if __name__=='__main__': unittest.main()
