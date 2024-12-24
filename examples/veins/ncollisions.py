EXPERIMENT = 'FMNIST_wscc'
METHOD = 'HVCFLThreshold'
ENVIRONMENT = 'final/8RSU/'
RESULTS_PATH = 'python/experiments/' + EXPERIMENT + '/' + ENVIRONMENT + METHOD + '/1/results/General-#0.sca'
NCOLLISIONS = 'ncollisions'
GENERATEDWSMS = 'generatedWSMs'

ncollisions = 0
total = 0
with open(RESULTS_PATH, 'r') as results_file:
    for line in results_file.readlines():
        if GENERATEDWSMS in line:
            total += int(line.split()[3])
        elif NCOLLISIONS in line:
            ncollisions += int(line.split()[3])

print('Number of Collisions: ', ncollisions)
print('Total Messages Generated: ', total)
print('Percentage of Collisions: ', ncollisions / float(total))