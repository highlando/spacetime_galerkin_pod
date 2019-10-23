import unittest
from get_gaussqr_wrtdnsty import get_gaussqr_uniform
import numpy as np


class GaussQRTests(unittest.TestCase):
    def setUp(self):
        self.Nlist = [1, 2, 3]
        self.absclist = [[.5],
                         [.5*(1-np.sqrt(1./3)), .5*(1+np.sqrt(1./3))],
                         [.5*(1-np.sqrt(3./5)), .5, .5*(1+np.sqrt(3./5))]]
        self.wghtlist = [[1],
                         [.5, .5],
                         [5./18, 8./18, 5./18]]

    def test_gaussqr_uniform(self):
        for k, N in enumerate(self.Nlist):
            abscissae, weights = get_gaussqr_uniform(N)
            # print(abscissae)
            # print(self.absclist[k])
            # print(weights)
            self.assertTrue(np.allclose(abscissae, np.array(self.absclist[k])))
            self.assertTrue(np.allclose(weights, self.wghtlist[k]))
