import unittest
import numpy as np
from chempath import Chempath

def get_chempath(input_folder, ignored_sb=[]):
    '''Gets a chempath object given an input folder and a list of species
    of interest'''
    chempath = Chempath(
        reactions_path=f'{input_folder}/reactions.txt',
        rates_path=f'{input_folder}/rates.dat',
        species_path=f'{input_folder}/species.txt',
        conc_path=f'{input_folder}/concentrations.dat',
        time_path=f'{input_folder}/model_time.dat',
        f_min=0, 
        dtype=np.float128,
        ignored_sb = ignored_sb
    )
    return chempath

def do_a_find_pathways_iteration(chempath, split_into_subpathways=False):
    '''Does a single iteration to find new pathways'''
    sb = chempath.get_sb()
    chempath.get_prod_destr_idxs(sb)
    chempath.form_new_pathways()
    chempath.calculate_deleted_pathways_effect()
    chempath.calculate_rates_explaining_conc_change()
    chempath.delete_old_pathways()
    chempath.delete_insignificant_pathways()
    if split_into_subpathways:
        chempath.split_into_subpathways()

class Chempath_Tests(unittest.TestCase):

    def test_get_sb(self):
        path='input/simple_ozone'
        chempath = get_chempath(path, ignored_sb=['O2'])

        # get first branching point
        sb = chempath.get_sb()
        self.assertEqual(sb, 'O')

        # get 2nd branching point
        sb = chempath.get_sb()
        self.assertEqual(sb, 'O3')

    def test_simple_ozone_find_all_pathways(self):
        path='input/simple_ozone'
        chempath = get_chempath(path, ignored_sb=['O2'])

        chempath.find_all_pathways()
        
        expected_xjk = np.array([[1., 0., 0.],
                                [0., 1., 1.],
                                [1., 2., 1.],
                                [0., 0., 1.]])
        expected_fk = np.array([80.,  9.,  1.])
        self.assertTrue(np.all(chempath.xjk == expected_xjk))
        self.assertTrue(np.all(np.isclose(chempath.fk, expected_fk)))
        
    def test_get_sij(self):
        path='input/simple_ozone'
        chempath = get_chempath(path, ignored_sb=['O2'])

        expected_sij = np.array([[-1.,  0.,  1., -1.],
                                [ 1., -1., -1.,  2.],
                                [ 1.,  2., -1., -1.]])
        self.assertTrue(np.all(chempath.sij == expected_sij))

    def test_find_pathways_one_iteration(self):
        path='input/simple_ozone'
        chempath = get_chempath(path, ignored_sb=['O2'])

        do_a_find_pathways_iteration(chempath)

        expected_xjk = np.array([[1., 1., 0., 0.],
                                [0., 0., 1., 1.],
                                [1., 0., 2., 0.],
                                [0., 1., 0., 2.]])
        expected_fk = np.array([79.2,  0.8,  9.9,  0.1])
        expected_pids = np.array(['1*0,1*2', '1*0,1*3', '1*1,2*2',
            '1*1,2*3'])
        expected_mik = np.array([[ 0., -2.,  2., -2.],
                                [ 0.,  3., -3.,  3.],
                                [ 0.,  0.,  0.,  0.]])
        expected_pi = np.array([19.8,  2.7,  0. ])
        expected_di = np.array([ 1.8, 29.7,  0. ])
        self.assertTrue(np.all(np.isclose(chempath.xjk, expected_xjk)))
        self.assertTrue(np.all(np.isclose(chempath.fk, expected_fk)))
        self.assertTrue(np.all(chempath.pathway_ids == expected_pids))
        self.assertTrue(np.all(np.isclose(chempath.mik, expected_mik)))
        self.assertTrue(np.all(np.isclose(chempath.pi, expected_pi)))
        self.assertTrue(np.all(np.isclose(chempath.di, expected_di)))

    def test_is_simple_pathway(self):
        path='input/simple_ozone'
        chempath = get_chempath(path, ignored_sb=['O2'])
        xjk_test = np.array([[1., 1., 0., 0.],
                            [0., 0., 1., 1.],
                            [1., 0., 1., 0.],
                            [0., 1., 0., 1.]])
        self.assertTrue(chempath.is_elementary_pathway(xjk_test, 2, 3))
        self.assertTrue(chempath.is_elementary_pathway(xjk_test, 3, 2))
        self.assertFalse(chempath.is_elementary_pathway(xjk_test, 1, 2))
        self.assertFalse(chempath.is_elementary_pathway(xjk_test, 2, 1))

    def test_find_elementary_pathways(self):
        path='input/simple_ozone'
        chempath = get_chempath(path, ignored_sb=['O2'])

        do_a_find_pathways_iteration(chempath)
        do_a_find_pathways_iteration(chempath)

        expected_el_pathways1 = np.array([[1., 0.],
                                        [0., 1.],
                                        [1., 1.],
                                        [0., 1.]])
        expected_el_pathways2 = np.array([[0.],
                                        [1.],
                                        [1.],
                                        [1.]])
        self.assertTrue(np.all(
            chempath.find_elementary_pathways(
                chempath.xjk[:,2], 2) == expected_el_pathways1)
            )
        self.assertTrue(np.all(
            chempath.find_elementary_pathways(
                chempath.xjk[:,3], 3) == expected_el_pathways2)
            )
         
        
    def test_delete_pathways(self):
        path='input/simple_ozone'
        chempath = get_chempath(path, ignored_sb=['O2'])
        chempath.f_min = 0.2

        do_a_find_pathways_iteration(chempath)

        expected_rj_del = np.array([0. , 0.1, 0. , 0.2])
        expected_di_del = np.array([0.2, 0. , 0. ])
        expected_pi_del = np.array([0. , 0.3, 0. ])
        self.assertTrue(np.all(np.isclose(chempath.rj_del, expected_rj_del)))
        self.assertTrue(np.all(np.isclose(chempath.pi_del, expected_pi_del)))
        self.assertTrue(np.all(np.isclose(chempath.di_del, expected_di_del)))

        chempath = get_chempath(path, ignored_sb=['O2'])
        chempath.f_min = 10
        chempath.find_all_pathways()

        expected_xjk = np.array([[1.],
                                [0.],
                                [1.],
                                [0.]])
        expected_fk = np.array([79.2])
        expected_rj_del = np.array([ 0.8, 10. , 19.8,  1.])
        expected_pi_del = np.array([19.8,  2.7,  0. ])
        expected_di_del = np.array([ 1.8, 29.7,  0. ])

        self.assertTrue(np.all(np.isclose(chempath.xjk, expected_xjk)))
        self.assertTrue(np.all(np.isclose(chempath.fk, expected_fk)))
        self.assertTrue(np.all(np.isclose(chempath.rj_del, expected_rj_del)))
        self.assertTrue(np.all(np.isclose(chempath.pi_del, expected_pi_del)))
        self.assertTrue(np.all(np.isclose(chempath.di_del, expected_di_del)))

    def test_simple_ozone_fmin_02(self):
        path='input/simple_ozone'
        chempath = get_chempath(path, ignored_sb=['O2'])
        chempath.f_min = 0.2
        chempath.find_all_pathways()

        expected_xjk = np.array([[1., 0., 0.],
                                [0., 1., 1.],
                                [1., 2., 1.],
                                [0., 0., 1.]])
        expected_fk = np.array([80. ,  9. ,  0.8])
        expected_rj_del = np.array([0. , 0.2, 0.2, 0.2])
        expected_pi_del = np.array([0.2, 0.3, 0. ])
        expected_di_del = np.array([0.2, 0.3, 0. ])

        self.assertTrue(np.all(np.isclose(chempath.xjk, expected_xjk)))
        self.assertTrue(np.all(np.isclose(chempath.fk, expected_fk)))
        self.assertTrue(np.all(np.isclose(chempath.rj_del, expected_rj_del)))
        self.assertTrue(np.all(np.isclose(chempath.pi_del, expected_pi_del)))
        self.assertTrue(np.all(np.isclose(chempath.di_del, expected_di_del)))



unittest.main()