import solver as sv

def main():
    mysolver = sv.MaxEntSolver()
    print 'Beta ' + str(mysolver.beta) + ' Tau ' + str(mysolver.tau)

    mysolver.setprobparams(1.0e-3, 1.0e-2)
    print 'Beta ' + str(mysolver.beta) + ' Tau ' + str(mysolver.tau)

    exit()


if __name__ == '__main__':
    main()