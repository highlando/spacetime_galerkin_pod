import unittest
from spacetime_galerkin_pod.chaos_expansion_utils import get_gaussqr_uniform
import numpy as np


class GaussQRTests(unittest.TestCase):
    def setUp(self):
        self.Nlist = [1, 2, 3, 4]
        self.absclist = [[.5],
                         [.5*(1-np.sqrt(1./3)), .5*(1+np.sqrt(1./3))],
                         [.5*(1-np.sqrt(3./5)), .5, .5*(1+np.sqrt(3./5))],
                         [.5*(1-np.sqrt((15+2*np.sqrt(30))/35)),
                          .5*(1-np.sqrt((15-2*np.sqrt(30))/35)),
                          .5*(1+np.sqrt((15-2*np.sqrt(30))/35)),
                          .5*(1+np.sqrt((15+2*np.sqrt(30))/35))]]

        self.wghtlist = [[1],
                         [.5, .5],
                         [5./18, 8./18, 5./18],
                         [(18-np.sqrt(30))/72, (18+np.sqrt(30))/72,
                          (18+np.sqrt(30))/72, (18-np.sqrt(30))/72]]

        self.absclistmoo = [[0.],
                            [-np.sqrt(1./3), np.sqrt(1./3)],
                            [-np.sqrt(3./5), 0., np.sqrt(3./5)],
                            [-np.sqrt((15+2*np.sqrt(30))/35),
                             -np.sqrt((15-2*np.sqrt(30))/35),
                             +np.sqrt((15-2*np.sqrt(30))/35),
                             +np.sqrt((15+2*np.sqrt(30))/35)]]

        self.wghtlistmoo = [[2.],
                            [1., 1.],
                            [5./9, 8./9, 5./9],
                            [(18-np.sqrt(30))/36, (18+np.sqrt(30))/36,
                             (18+np.sqrt(30))/36, (18-np.sqrt(30))/36]]

    def test_gaussqr_uniform_zo(self):
        '''
        testing the gauss qr points/weights for (0,1)
        '''

        for k, N in enumerate(self.Nlist):
            abscissae, weights = get_gaussqr_uniform(N)
            print(abscissae)
            print(self.absclist[k])
            print(weights)
            print(self.wghtlist[k])
            self.assertTrue(np.allclose(abscissae, np.array(self.absclist[k])))
            self.assertTrue(np.allclose(weights, self.wghtlist[k]))

    def test_gaussqr_uniform_moo(self):
        '''
        testing the gauss qr points/weights for (-1,1)
        '''

        for k, N in enumerate(self.Nlist):
            abscissae, weights = get_gaussqr_uniform(N, a=-1., b=1.)
            print('k=', k)
            print(abscissae)
            print(self.absclistmoo[k])
            print(weights)
            print(self.wghtlistmoo[k])
            self.assertTrue(np.allclose(abscissae,
                                        np.array(self.absclistmoo[k])))
            self.assertTrue(np.allclose(weights, self.wghtlistmoo[k]))

if __name__ == '__main__':
    unittest.main()
