import multiprocessing
import numpy as np
import nn

def run_main(arg):
    nn.main(arg)

if __name__ == '__main__':

    arguments = np.array([
        "validation_one('Estar', ['TI33_750a'], 'TI33_750a', 'Exp', [20])"
        ])
    
    processes = []
    num_processes = len(arguments)
    for i in range(num_processes):        
        process = multiprocessing.Process(target=run_main, args=(arguments[i],))
        processes.append(process)

    for process in processes:
        process.start()
    for process in processes:
        process.join()


        '''
        "validation_one('sigma_y', ['TI33_750a'], 'TI33_750a', 'Exp', [1])",
        "validation_one('sigma_y', ['TI33_750a'], 'TI33_750a', 'Exp', [2])",
        "validation_one('sigma_y', ['TI33_750a'], 'TI33_750a', 'Exp', [3])",
        "validation_one('sigma_y', ['TI33_750a'], 'TI33_750a', 'Exp', [4])",
        "validation_one('sigma_y', ['TI33_750a'], 'TI33_750a', 'Exp', [5])",
        "validation_one('sigma_y', ['TI33_750a'], 'TI33_750a', 'Exp', [10])",
        "validation_one('sigma_y', ['TI33_750a'], 'TI33_750a', 'Exp', [20])",
        "validation_one('Estar', ['TI33_750a'], 'TI33_750a', 'Exp', [1])",
        "validation_one('Estar', ['TI33_750a'], 'TI33_750a', 'Exp', [2])",
        "validation_one('Estar', ['TI33_750a'], 'TI33_750a', 'Exp', [3])",
        "validation_one('Estar', ['TI33_750a'], 'TI33_750a', 'Exp', [4])",
        "validation_one('Estar', ['TI33_750a'], 'TI33_750a', 'Exp', [5])",
        "validation_one('Estar', ['TI33_750a'], 'TI33_750a', 'Exp', [10])",'''
        '''"validation_one('sigma_y', ['TI33_25a', 'TI33_250a', 'TI33_500a'], 'TI33_750a', 'Exp', [1, 1, 1])",
        "validation_one('sigma_y', ['TI33_25a', 'TI33_250a', 'TI33_500a'], 'TI33_750a', 'Exp', [2, 2, 2])",
        "validation_one('sigma_y', ['TI33_25a', 'TI33_250a', 'TI33_500a'], 'TI33_750a', 'Exp', [3, 3, 3])",
        "validation_one('sigma_y', ['TI33_25a', 'TI33_250a', 'TI33_500a'], 'TI33_750a', 'Exp', [4, 4, 4])",
        "validation_one('sigma_y', ['TI33_25a', 'TI33_250a', 'TI33_500a'], 'TI33_750a', 'Exp', [5, 5, 5])",
        "validation_one('sigma_y', ['TI33_25a', 'TI33_250a', 'TI33_500a'], 'TI33_750a', 'Exp', [10, 10, 10])",
        "validation_one('sigma_y', ['TI33_25a', 'TI33_250a', 'TI33_500a'], 'TI33_750a', 'Exp', [20, 20, 20])",
        "validation_one('Estar', ['TI33_25a', 'TI33_250a', 'TI33_500a'], 'TI33_750a', 'Exp', [1, 1, 1])",
        "validation_one('Estar', ['TI33_25a', 'TI33_250a', 'TI33_500a'], 'TI33_750a', 'Exp', [2, 2, 2])",
        "validation_one('Estar', ['TI33_25a', 'TI33_250a', 'TI33_500a'], 'TI33_750a', 'Exp', [3, 3, 3])",
        "validation_one('Estar', ['TI33_25a', 'TI33_250a', 'TI33_500a'], 'TI33_750a', 'Exp', [4, 4, 4])",
        "validation_one('Estar', ['TI33_25a', 'TI33_250a', 'TI33_500a'], 'TI33_750a', 'Exp', [5, 5, 5])",
        "validation_one('Estar', ['TI33_25a', 'TI33_250a', 'TI33_500a'], 'TI33_750a', 'Exp', [10, 10, 10])",
        "validation_one('Estar', ['TI33_25a', 'TI33_250a', 'TI33_500a'], 'TI33_750a', 'Exp', [20, 20, 20])"'''