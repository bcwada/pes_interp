import numpy as np
import unittests
from pathlib import Path

import sheppard_pes as sheppard
import lib.xyz as xyz

class sheppard_test(unittest.TestCase):

    def pseudo(self, coords):
        ref = sheppard.Pes_Point.calc_inv_dist(g.coords.reshape(-1))
        z = sheppard.Pes_Point.calc_inv_dist(coords)
        v = z - ref
        return 0.5 * np.dot(v,v)

    def point_test(self, test_path, pes):
        test_geom = xyz.Geometry.from_file(test_path)
        pseudo_e = self.pseudo(test_geom.coords.reshape(-1))
        print(f"evaluated pseudo at test: {pseudo_e}")
        e = pes.eval_point(test_geom.coords.reshape(-1))
        print(f"evaluated sheppard at test: {e}")
        self.assertAlmostEqual(pseudo_e, e, m)

    def setUp(self):
        path = Path("./test/sheppard_pes/BuH.xyz")
        g = xyz.Geometry.from_file(path)
        pes = sheppard.Pes.new_pes()

    def test_test(self):
        self.assertEqual(1,1)



if __name__ == "__main__":
    unittest.main()