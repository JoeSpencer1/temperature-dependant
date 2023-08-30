import multiprocessing
import numpy as np
import nn

def run_main(arg):
    nn.main(arg)

if __name__ == '__main__':

    arguments = np.array([
        "validation_mod_exp('Estar', 0, 'B3090_25')"
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

''',
        "validation_mod_exp('Estar', 0, 'B3090_250')",
        "validation_mod_exp('Estar', 0, 'B3090_500')",
        "validation_mod_exp('Estar', 0, 'B3090_750')"'''

'''
        "validation_mod_exp('Estar', 1, 'B3090_25')",
        "validation_mod_exp('Estar', 2, 'B3090_25')",
        "validation_mod_exp('Estar', 3, 'B3090_25')",
        "validation_mod_exp('Estar', 4, 'B3090_25')",
        "validation_mod_exp('Estar', 5, 'B3090_25')",
        "validation_mod_exp('Estar', 6, 'B3090_25')",
        "validation_mod_exp('Estar', 8, 'B3090_25')",
        "validation_mod_exp('Estar', 10, 'B3090_25')",
        "validation_mod_exp('Estar', 20, 'B3090_25')",
        "validation_mod_exp('Estar', 1, 'B3090_250')",
        "validation_mod_exp('Estar', 2, 'B3090_250')",
        "validation_mod_exp('Estar', 3, 'B3090_250')",
        "validation_mod_exp('Estar', 4, 'B3090_250')",
        "validation_mod_exp('Estar', 5, 'B3090_250')",
        "validation_mod_exp('Estar', 6, 'B3090_250')",
        "validation_mod_exp('Estar', 8, 'B3090_250')",
        "validation_mod_exp('Estar', 10, 'B3090_250')",
        "validation_mod_exp('Estar', 20, 'B3090_250')",
        "validation_mod_exp('Estar', 1, 'B3090_500')",
        "validation_mod_exp('Estar', 2, 'B3090_500')",
        "validation_mod_exp('Estar', 3, 'B3090_500')",
        "validation_mod_exp('Estar', 4, 'B3090_500')",
        "validation_mod_exp('Estar', 5, 'B3090_500')",
        "validation_mod_exp('Estar', 6, 'B3090_500')",
        "validation_mod_exp('Estar', 8, 'B3090_500')",
        "validation_mod_exp('Estar', 10, 'B3090_500')",
        "validation_mod_exp('Estar', 20, 'B3090_500')",
        "validation_mod_exp('Estar', 1, 'B3090_750')",
        "validation_mod_exp('Estar', 2, 'B3090_750')",
        "validation_mod_exp('Estar', 3, 'B3090_750')",
        "validation_mod_exp('Estar', 4, 'B3090_750')",
        "validation_mod_exp('Estar', 5, 'B3090_750')",
        "validation_mod_exp('Estar', 6, 'B3090_750')",
        "validation_mod_exp('Estar', 8, 'B3090_750')",
        "validation_mod_exp('Estar', 10, 'B3090_750')",
        "validation_mod_exp('Estar', 20, 'B3090_750')"
        '''