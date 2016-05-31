from SimPEG import *


def run(N=100, plotIt=True):
    """
        Inversion: Linear Problem
        =========================

        Here we go over the basics of creating a linear problem and inversion.

    """


    np.random.seed(1)

    std_noise = 1e-2

    mesh = Mesh.TensorMesh([N])

    m0 = np.ones(mesh.nC) * 1e-4
    mref = np.zeros(mesh.nC)

    nk = 10
    jk = np.linspace(1.,nk,nk)
    p = -2.
    q = 1.

    g = lambda k: np.exp(p*jk[k]*mesh.vectorCCx)*np.cos(np.pi*q*jk[k]*mesh.vectorCCx)

    G = np.empty((nk, mesh.nC))

    for i in range(nk):
        G[i,:] = g(i)

    mtrue = np.zeros(mesh.nC)
    mtrue[mesh.vectorCCx > 0.3] = 1.
    mtrue[mesh.vectorCCx > 0.45] = -0.5
    mtrue[mesh.vectorCCx > 0.6] = 0


    prob = Problem.LinearProblem(mesh, G)
    survey = Survey.LinearSurvey()
    survey.pair(prob)
    survey.dobs = prob.fields(mtrue) + std_noise * np.random.randn(nk)
    #survey.makeSyntheticData(mtrue, std=std_noise)

    wd = np.ones(nk) * std_noise

    #print survey.std[0]
    #M = prob.mesh
    # Distance weighting
    wr = np.sum(prob.G**2.,axis=0)**0.5
    wr = ( wr/np.max(wr) )

#    reg = Regularization.Simple(mesh)
#    reg.mref = mref
#    reg.cell_weights = wr
#
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.Wd = 1./wd
#
#    opt = Optimization.ProjectedGNCG(maxIter=20,lower=-2.,upper=2., maxIterCG= 10, tolCG = 1e-4)
#    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
#    invProb.curModel = m0
#
#    beta = Directives.BetaSchedule(coolingFactor=2, coolingRate=1)
#    target = Directives.TargetMisfit()
#
    betaest = Directives.BetaEstimate_ByEig()
#    inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaest, target])
#
#
#    mrec = inv.run(m0)
#    ml2 = mrec
#    print "Final misfit:" + str(invProb.dmisfit.eval(mrec))
#
#    # Switch regularization to sparse
#    phim = invProb.phi_m_last
#    phid =  invProb.phi_d

    reg = Regularization.Sparse(mesh)
    reg.mref = mref
    reg.cell_weights = wr

    reg.mref = np.zeros(mesh.nC)
    eps_p = 5e-2
    eps_q = 5e-2
    norms   = [0., 0., 2., 2.]

    opt = Optimization.ProjectedGNCG(maxIter=100 ,lower=-2.,upper=2., maxIterLS = 20, maxIterCG= 10, tolCG = 1e-3)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    update_Jacobi = Directives.Update_lin_PreCond()
    IRLS = Directives.Update_IRLS( norms=norms,  eps_p=eps_p, eps_q=eps_q)

    inv = Inversion.BaseInversion(invProb, directiveList=[IRLS,betaest,update_Jacobi])

    # Run inversion
    mrec = inv.run(m0)

    print "Final misfit:" + str(invProb.dmisfit.eval(mrec))


    if plotIt:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1,2,figsize=(12*1.2,4*1.2))
        for i in range(prob.G.shape[0]):
            axes[0].plot(prob.G[i,:])
        axes[0].set_title('Columns of matrix G')

        axes[1].plot(mesh.vectorCCx, mtrue, 'b-')
        axes[1].plot(mesh.vectorCCx, reg.l2model, 'r-')
        #axes[1].legend(('True Model', 'Recovered Model'))
        axes[1].set_ylim(-1.0,1.25)

        axes[1].plot(mesh.vectorCCx, mrec, 'k-',lw = 2)
        axes[1].legend(('True Model', 'Smooth l2-l2',
        'Sparse lp:' + str(reg.norms[0]) + ', lqx:' + str(reg.norms[1]) ), fontsize = 12)
        plt.show()

    return prob, survey, mesh, mrec

if __name__ == '__main__':
    run()
