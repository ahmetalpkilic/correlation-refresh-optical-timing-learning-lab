import unittest
import numpy as np
import lab as l
import repeat_model as r
import clock_model as c

class RefreshTests(unittest.TestCase):
    def test_lifetime_cost_and_no_change_control(self):
        for n,m in l.POLICIES.values(): self.assertEqual(160+2*n+40*m,640)
        for m in (8,10,12):
            for rho in (-.6,0.,.4,.9):self.assertAlmostEqual(l.transfer_factor(m,rho,rho),1.,places=12)

    def test_transfer_bias_independent_recursion(self):
        # Simulate the regime after the change using an independent recurrence.
        rng=np.random.default_rng(812);m=12;actual=.9;assumed=.4
        z=rng.normal(size=(15000,m,2))
        for j in range(1,m):z[:,j]=actual*z[:,j-1]+np.sqrt(1-actual**2)*z[:,j]
        q=r.residual_precision(m,assumed)
        observed=np.einsum('tki,kl,tlj->ij',z,q,z)/(len(z)*(m-1))
        np.testing.assert_allclose(observed,l.transfer_factor(m,assumed,actual)*np.eye(2),atol=.012)
        self.assertLess(l.transfer_factor(m,assumed,actual),.5)

    def test_policy_data_boundaries(self):
        rng=np.random.default_rng(6);z=rng.normal(size=(40,12,2))
        old=np.array([.1,.4]); fresh={40:np.array([.2,.7]),80:np.array([.3,.8])}
        for name,(n,m) in l.POLICIES.items():
            a=l.estimate_policy(z,old,fresh,name)
            changed=z.copy();changed[:,m:]+=1000
            np.testing.assert_array_equal(a,l.estimate_policy(changed,old,fresh,name))
            selected=old if n==0 else fresh[n]
            np.testing.assert_array_equal(a,r.estimate(z[:,:m],selected[r.LABELS]))
        np.testing.assert_array_equal(l.estimate_policy(z,old,fresh,'stale'),l.estimate_policy(z,old,{40:old,80:old},'stale'))
        np.testing.assert_array_equal(l.estimate_policy(z,old,fresh,'refresh40'),l.estimate_policy(z,old+.2,fresh,'refresh40'))

    def test_common_covariance_scale_cancels_clock_fit(self):
        rng=np.random.default_rng(24);cov=c.covariances(4.)
        z=np.column_stack((c.SYNC,c.RATE*c.SYNC+c.OFFSET))+np.einsum('nij,nj->ni',np.linalg.cholesky(cov),rng.normal(size=(40,2)))
        np.testing.assert_allclose(c.calibrate(*z.T,cov),c.calibrate(*z.T,3*cov),atol=1e-7)

    def test_fresh_seed_end_to_end_and_determinism(self):
        a=l.experiment(trials=3,seed=117);b=l.experiment(trials=3,seed=117)
        self.assertEqual(a,b);self.assertEqual(a['failed_fits'],0)
        for row in a['rows']:
            self.assertEqual(set(row['methods']),{'oracle','stale','refresh40','refresh80','known12','known10','known8'})
            for stats in row['methods'].values():self.assertTrue(np.isfinite(stats['rmse_ms']))
            for stats in row['rho_estimates'].values():self.assertTrue(np.all(np.abs(stats['mean'])<=.95))
        with self.assertRaises(ValueError):l.experiment(trials=1)

if __name__=='__main__':unittest.main()
